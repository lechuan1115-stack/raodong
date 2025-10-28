#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, time, json, csv, argparse, datetime, os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mydata_read
import mymodel1
from utils import AverageMeter
from pytorchtools import EarlyStopping


# ------------------------------
# 统计 z/s 的分布（辅助诊断）
# ------------------------------
@torch.no_grad()
def summarize_zs(loader, tag="val", max_batches=50):
    try:
        from gconv import PERTURB_ORDER
    except Exception:
        PERTURB_ORDER = ['CFO','SCALE','GAIN','SHIFT','CHIRP']

    tot = 0
    act_sum = None
    s_sum = None
    s_sqsum = None

    for bi, batch in enumerate(loader):
        if len(batch) < 4:
            print(f"[Z/S Summary-{tag}] 此数据集 batch 不含 z/s 字段（len={len(batch)}），无法统计")
            return
        z = torch.as_tensor(batch[2])
        s = torch.as_tensor(batch[3])
        m = (z > 0.5).float()

        if act_sum is None:
            act_sum = m.sum(0)
            s_sum   = torch.nan_to_num(s).sum(0)
            s_sqsum = (torch.nan_to_num(s) ** 2).sum(0)
        else:
            act_sum += m.sum(0)
            s_sum   += torch.nan_to_num(s).sum(0)
            s_sqsum += (torch.nan_to_num(s) ** 2).sum(0)
        tot += z.size(0)
        if bi + 1 >= max_batches: break

    if tot == 0:
        print(f"[Z/S Summary-{tag}] loader 为空"); return

    act_rate = (act_sum / tot).tolist()
    s_mean   = (s_sum / tot).tolist()
    s_std    = ((s_sqsum / tot) - (s_sum / tot) ** 2).clamp(min=0).sqrt().tolist()

    def _fmt(vs): return {k: float(v) for k, v in zip(PERTURB_ORDER, vs)}
    print(f"[Z/S Summary-{tag}] act_rate:", _fmt(act_rate))
    print(f"[Z/S Summary-{tag}] s_mean  :", _fmt(s_mean))
    print(f"[Z/S Summary-{tag}] s_std   :", _fmt(s_std))


# ------------------------------
# 先验操控：zero/shuffle + (新增) prior_drop / s_jitter
# ------------------------------
def _maybe_manipulate_zs(z, s, args):
    if z is None and s is None:
        return z, s

    if z is not None: z = z.clone()
    if s is not None: s = s.clone()

    # 评估用开关
    if z is not None:
        if args.zero_z:
            z[:] = 0
        elif args.shuffle_z:
            idx = torch.randperm(z.size(0), device=z.device)
            z = z[idx]
    if s is not None and args.zero_s:
        s[:] = 0

    # 削弱先验：随机丢弃 + s 抖动（仅在训练/eval统一调用处执行）
    prior_drop = float(getattr(args, 'prior_drop', 0.0))
    s_jitter   = float(getattr(args, 's_jitter', 0.0))
    if z is not None and s is not None and prior_drop > 0:
        drop = torch.bernoulli(torch.full_like(z, prior_drop))
        # 中性值顺序：['CFO','SCALE','GAIN','SHIFT','CHIRP']
        neutral = torch.tensor([0., 1., 1., 0., 0.], device=z.device, dtype=s.dtype)
        s = torch.where(drop > 0, neutral.unsqueeze(0).expand_as(s), s)
        z = torch.where(drop > 0, torch.zeros_like(z), z)

    if z is not None and s is not None and s_jitter > 0:
        # 只对激活维度做乘性抖动：s <- s * (1 + N(0, s_jitter))
        noise = torch.randn_like(s) * s_jitter
        m = (z > 0.5).float()
        s = s * (1.0 + noise) * m + s * (1.0 - m)

    return z, s


# ---- Mixup ----
def _mixup(x, y, alpha=0.2):
    if alpha is None or alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return x_mix, y_a, y_b, perm, lam


def _set_steer_disabled(model, flag: bool):
    if hasattr(model, "set_steer_disabled"):
        model.set_steer_disabled(flag)


def _set_first_debug(model, flag: bool):
    if hasattr(model, "set_first_debug"):
        model.set_first_debug(flag)


def _forward(model, batch, device, args, want_debug=False):
    if len(batch) == 2:
        x, y = batch; z = None; s = None
    elif len(batch) == 3:
        x, y, z = batch; s = None
    else:
        x, y, z, s = batch[0], batch[1], batch[2], batch[3]

    x = x.float().to(device, non_blocking=True)
    y = torch.as_tensor(y, device=device).long().view(-1)
    if z is not None: z = torch.as_tensor(z, device=device)
    if s is not None: s = torch.as_tensor(s, device=device)

    z, s = _maybe_manipulate_zs(z, s, args)

    _set_first_debug(model, args.first_debug or want_debug)

    if (z is not None) and (s is not None):
        logits, feat = model(x, z, s)
    elif z is not None:
        logits, feat = model(x, z)
    else:
        logits, feat = model(x)

    dbg = {}
    if (args.first_debug or want_debug) and hasattr(model, 'get_first_last_debug'):
        dbg = model.get_first_last_debug() or {}
    return logits, feat, y, dbg


def train_one_epoch(model, criterion, optimizer, loader, device, args):
    model.train()
    losses = AverageMeter()
    correct = total = 0

    for i, batch in enumerate(loader):
        _set_steer_disabled(model, False)

        # 取数据张量
        if len(batch) == 2:
            x, y = batch; z = None; s = None
        elif len(batch) == 3:
            x, y, z = batch; s = None
        else:
            x, y, z, s = batch[0], batch[1], batch[2], batch[3]

        x = x.float().to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device).long().view(-1)
        if z is not None: z = torch.as_tensor(z, device=device)
        if s is not None: s = torch.as_tensor(s, device=device)
        z, s = _maybe_manipulate_zs(z, s, args)

        # Mixup
        x_mix, y_a, y_b, _, lam = _mixup(x, y, alpha=args.mixup_alpha)

        if (z is not None) and (s is not None):
            logits, feat = model(x_mix, z, s)
        elif z is not None:
            logits, feat = model(x_mix, z)
        else:
            logits, feat = model(x_mix)

        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b) if y_b is not None else criterion(logits, y)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        losses.update(loss.item(), y.size(0))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)

        if (i + 1) % args.print_freq == 0:
            print(f"  iter {i+1:04d} | loss {losses.avg:.4f}")

    acc = 100.0 * correct / max(1, total)
    return acc, losses.avg


@torch.no_grad()
def evaluate(model, criterion, loader, device, args, mode="normal", print_debug=False):
    model.eval()

    bak_shuffle, bak_zeroz, bak_zeros = args.shuffle_z, args.zero_z, args.zero_s
    try:
        if mode == "no_steer":
            _set_steer_disabled(model, True)
            args.shuffle_z, args.zero_z, args.zero_s = False, False, False
        elif mode == "shuffle_z":
            _set_steer_disabled(model, False)
            args.shuffle_z, args.zero_z, args.zero_s = True, False, False
        else:
            _set_steer_disabled(model, False)
            args.shuffle_z, args.zero_z, args.zero_s = False, False, False

        print(f"[EVAL MODE] {mode} | steer_disabled={'True' if (mode=='no_steer') else 'False'}")

        losses = AverageMeter()
        correct = total = 0
        last_dbg = {}

        for batch in loader:
            logits, feat, y, dbg = _forward(model, batch, device, args, want_debug=print_debug)
            loss = criterion(logits, y)
            losses.update(loss.item(), y.size(0))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
            if dbg: last_dbg = dbg

        acc = 100.0 * correct / max(1, total)
        if print_debug and last_dbg:
            print("[FirstLayer Debug]", last_dbg)
    finally:
        args.shuffle_z, args.zero_z, args.zero_s = bak_shuffle, bak_zeroz, bak_zeros
        _set_steer_disabled(model, False)

    return acc, losses.avg


@torch.no_grad()
def test_metrics(model, loader, device, args):
    model.eval()
    total = 0
    top1 = top2 = top3 = top5 = 0.0
    for batch in loader:
        logits, feat, y, _ = _forward(model, batch, device, args, want_debug=False)
        maxk = 5
        _, topk = logits.topk(maxk, 1, True, True)
        topk = topk.t()
        corr = topk.eq(y.view(1, -1).expand_as(topk))
        top1 += corr[:1].reshape(-1).float().sum().item()
        top2 += corr[:2].any(0).float().sum().item()
        top3 += corr[:3].any(0).float().sum().item()
        top5 += corr[:5].any(0).float().sum().item()
        total += y.size(0)
    return dict(
        top1=top1/total*100.0, top2=top2/total*100.0,
        top3=top3/total*100.0, top5=top5/total*100.0
    )


def run_one_model(args, model_name, data_path, save_dir, device):
    print("="*80)
    print(f"模型：{model_name}")
    print(f"数据：{data_path}")
    print(f"保存：{save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # 数据
    dataset = mydata_read.SignalDataset1(data_path, shuffle=False, seed=args.seed)
    fs = getattr(dataset, 'fs', 50e6)

    # --- 分层划分：80/10/10 ---
    ys = torch.as_tensor(dataset.Y).long().tolist()
    from collections import defaultdict, Counter
    by_cls = defaultdict(list)
    for i, y in enumerate(ys):
        by_cls[y].append(i)

    tr_idx, va_idx, te_idx = [], [], []
    rng = np.random.RandomState(args.seed)
    for cls, ids in by_cls.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(0.8 * n))
        n_va = int(round(0.1 * n))
        n_te = n - n_tr - n_va
        tr_idx.extend(ids[:n_tr])
        va_idx.extend(ids[n_tr:n_tr + n_va])
        te_idx.extend(ids[n_tr + n_va:])

    from torch.utils.data import Subset
    train_set   = Subset(dataset, tr_idx)
    validate_set= Subset(dataset, va_idx)
    test_set    = Subset(dataset, te_idx)

    pin = device.type == 'cuda'
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)
    valloader   = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)
    testloader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)

    # 看看 z/s 是否正常
    summarize_zs(trainloader, tag="train", max_batches=50)
    summarize_zs(valloader,  tag="val",   max_batches=50)

    # 模型（根据类型选择参数）
    if model_name == 'p4all':
        model = mymodel1.create(
            name=model_name,
            num_classes=args.class_num,
            fs=fs,
            debug_first=args.first_debug,
            coef_scale=args.coef_scale,
            coef_gain=args.coef_gain,
            coef_cfo=args.coef_cfo,
            coef_chirp=args.coef_chirp,
        )
    else:
        model = mymodel1.create(name=model_name, num_classes=args.class_num)
    model = model.to(device)

    # ====== class_weight from train_set frequency ======
    num_classes = args.class_num
    from collections import Counter
    cnt = Counter([ys[i] for i in tr_idx])
    freq = np.array([cnt.get(c, 1) for c in range(num_classes)], dtype=np.float32)
    class_weight = 1.0 / np.maximum(freq, 1.0)
    class_weight = class_weight * (num_classes / class_weight.sum())
    class_weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)

    # 优化器 / 早停 / 调度
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    ckpt_path = osp.join(save_dir, f"{model_name}_best.pt")
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_path)

    # 训练循环
    t0 = time.time()
    history = []
    best_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    for epoch in range(args.max_epoch):
        print(f"==> Epoch {epoch+1}/{args.max_epoch}")
        a_tr, l_tr = train_one_epoch(model, criterion, optimizer, trainloader, device, args)
        a_ev, l_ev = evaluate(model, criterion, valloader, device, args, mode="normal", print_debug=args.first_debug)
        print(f"Train_Acc(%): {a_tr:.2f}  Eval_Acc(%): {a_ev:.2f}")
        print(f"Train_Loss: {l_tr:.4f}  Eval_Loss: {l_ev:.4f}")

        history.append({
            "epoch": int(epoch + 1),
            "train_acc": float(a_tr),
            "train_loss": float(l_tr),
            "val_acc": float(a_ev),
            "val_loss": float(l_ev),
        })

        if l_ev < best_val_loss:
            best_val_loss = float(l_ev)
            best_epoch = epoch + 1
        if a_ev > best_val_acc:
            best_val_acc = float(a_ev)

        early_stopping(l_ev, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

    if best_epoch == 0:
        best_epoch = len(history)

    elapsed = time.time() - t0
    print("训练耗时：", str(datetime.timedelta(seconds=round(elapsed))))

    # 测试 & 对照评估
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("####### 测试")
    _ , _ = evaluate(model, criterion, valloader, device, args, mode="normal", print_debug=True)

    print("== 对照评估（验证集）==")
    acc_n, _  = evaluate(model, criterion, valloader, device, args, mode='normal')
    acc_ns, _ = evaluate(model, criterion, valloader, device, args, mode='no_steer')
    acc_sz, _ = evaluate(model, criterion, valloader, device, args, mode='shuffle_z')
    print(f"[COMPARE] normal={acc_n:.2f}% | no_steer={acc_ns:.2f}% | shuffle_z={acc_sz:.2f}%")

    print("####### 最终测试集指标")
    test_stats = test_metrics(model, testloader, device, args)
    print(f"Top1:{test_stats['top1']:5.2f}%  Top2:{test_stats['top2']:5.2f}%  Top3:{test_stats['top3']:5.2f}%  Top5:{test_stats['top5']:5.2f}%")

    metrics = {
        "val_normal": float(acc_n),
        "val_no_steer": float(acc_ns),
        "val_shuffle_z": float(acc_sz),
        "test_top1": float(test_stats['top1']),
        "test_top2": float(test_stats['top2']),
        "test_top3": float(test_stats['top3']),
        "test_top5": float(test_stats['top5']),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "train_time_sec": float(elapsed),
    }

    # 写 CSV
    csv_path = osp.join(save_dir, "model_summary.csv")
    new_row = {
        **{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in metrics.items()},
        "model": model_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    fieldnames = ["model"] + [k for k in metrics.keys()] + ["timestamp"]
    write_header = (not osp.exists(csv_path))
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)
    print(f"[CSV] 结果已追加到: {csv_path}")

    summary = {
        "model": model_name,
        "metrics": metrics,
        "history": history,
        "save_dir": save_dir,
    }

    json_path = osp.join(save_dir, f"{model_name}_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 训练日志已保存到: {json_path}")

    return summary


def plot_model_comparison(results, output_path):
    if not results:
        return

    metric_keys = ["val_normal", "val_no_steer", "val_shuffle_z", "test_top1", "test_top2", "test_top3", "test_top5"]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(results), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, res in enumerate(results):
        values = [res['metrics'][k] for k in metric_keys]
        axes[0].bar(x + idx * width, values, width=width, label=res['model'])
    axes[0].set_xticks(x + width * (len(results) - 1) / 2)
    axes[0].set_xticklabels(metric_keys, rotation=30, ha='right')
    axes[0].set_ylabel('Accuracy / Top-k (%)')
    axes[0].set_title('验证与测试指标对比')
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.3)

    for res in results:
        epochs = [h['epoch'] for h in res['history']]
        if not epochs:
            continue
        train_acc = [h['train_acc'] for h in res['history']]
        val_acc = [h['val_acc'] for h in res['history']]
        axes[1].plot(epochs, train_acc, label=f"{res['model']} Train")
        axes[1].plot(epochs, val_acc, linestyle='--', label=f"{res['model']} Val")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('训练 / 验证准确率曲线')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend()

    fig.suptitle('模型性能对比', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("P4All vs CNN-Transformer benchmark")
    parser.add_argument('--data', type=str, default='./data/ADS-B_0dB_train.mat')
    parser.add_argument('--save-root', type=str, default='./runs_benchmark')
    parser.add_argument('--class_num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--first_debug', action='store_true')
    parser.add_argument('--print_freq', type=int, default=5, help='打印间隔（iteration）')

    # z/s 评估开关
    parser.add_argument('--shuffle_z', action='store_true')
    parser.add_argument('--zero_z', action='store_true')
    parser.add_argument('--zero_s', action='store_true')

    # 选择模型
    parser.add_argument('--models', type=str, default='p4all,cnn-transformer',
                        help='逗号分隔的模型列表，可选: p4all, cnn-transformer')

    # steer 强度与先验衰减
    parser.add_argument('--coef-scale', type=float, default=0.5)
    parser.add_argument('--coef-gain',  type=float, default=0.5)
    parser.add_argument('--coef-cfo',   type=float, default=0.5)
    parser.add_argument('--coef-chirp', type=float, default=0.5)
    parser.add_argument('--prior-drop', type=float, default=0.3,
                        help='随机丢弃先验 z/s 的概率 [0,1]')
    parser.add_argument('--s-jitter',   type=float, default=0.2,
                        help='对先验 s 的乘性噪声幅度，0为不抖动')

    # Mixup
    parser.add_argument('--mixup-alpha', type=float, default=0.2)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    torch.manual_seed(args.seed)
    if use_gpu: torch.cuda.manual_seed_all(args.seed)

    requested = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    valid_models = {'p4all', 'cnn-transformer'}
    models = []
    for m in requested:
        if m not in valid_models:
            raise ValueError(f"未知模型: {m}. 可选 {sorted(valid_models)}")
        if m not in models:
            models.append(m)

    print(f"程序：闭集识别（群作用验证，对比分析）")
    print(f"数据：{args.data}")
    print(f"模型列表：{models}")
    print(f"Steer系数: scale={args.coef_scale} gain={args.coef_gain} cfo={args.coef_cfo} chirp={args.coef_chirp} | prior_drop={args.prior_drop} s_jitter={args.s_jitter}")

    os.makedirs(args.save_root, exist_ok=True)

    all_results = []
    for m in models:
        save_dir = osp.join(args.save_root, m)
        result = run_one_model(args, m, args.data, save_dir, device)
        all_results.append(result)

    if len(all_results) >= 2:
        compare_path = osp.join(args.save_root, 'model_comparison.png')
        plot_model_comparison(all_results, compare_path)
        print(f"[PLOT] 模型对比图已生成: {compare_path}")

        metrics_csv = osp.join(args.save_root, 'model_comparison.csv')
        metric_keys = ["val_normal", "val_no_steer", "val_shuffle_z", "test_top1", "test_top2", "test_top3", "test_top5", "best_epoch", "best_val_loss", "best_val_acc", "train_time_sec"]
        write_header = (not osp.exists(metrics_csv))
        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model"] + metric_keys)
            if write_header:
                writer.writeheader()
            for res in all_results:
                row = {"model": res['model']}
                row.update({k: res['metrics'][k] for k in metric_keys})
                writer.writerow(row)
        print(f"[CSV] 模型对比指标已写入: {metrics_csv}")


if __name__ == '__main__':
    main()
