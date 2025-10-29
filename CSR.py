#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, time, json, csv, datetime, math, os.path as osp
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")  # 无图形界面环境下仍可绘图
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

        x_eval = x  # 留一份原始样本用于精度统计

        # Mixup
        x_mix, y_a, y_b, _, lam = _mixup(x, y, alpha=args.mixup_alpha)

        if (z is not None) and (s is not None):
            logits, _ = model(x_mix, z, s)
        elif z is not None:
            logits, _ = model(x_mix, z)
        else:
            logits, _ = model(x_mix)

        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b) if y_b is not None else criterion(logits, y)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        losses.update(loss.item(), y.size(0))

        # 以 eval 模式重新计算原始样本的精度，避免 Mixup 影响
        with torch.no_grad():
            prev_mode = model.training
            model.eval()
            if (z is not None) and (s is not None):
                logits_eval, _ = model(x_eval, z, s)
            elif z is not None:
                logits_eval, _ = model(x_eval, z)
            else:
                logits_eval, _ = model(x_eval)
            pred = logits_eval.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if prev_mode:
                model.train()

        if (i + 1) % args.print_freq == 0:
            print(f"  iter {i+1:04d} | loss {losses.avg:.4f}")

    acc = 100.0 * correct / max(1, total)
    return acc, losses.avg


@torch.no_grad()
def evaluate(model, criterion, loader, device, args, print_debug=False,
             max_batches=None, collect_outputs=False):
    model.eval()
    _set_steer_disabled(model, False)

    losses = AverageMeter()
    correct = total = 0
    last_dbg = {}
    logits_all = []
    labels_all = []
    preds_all = []

    for bi, batch in enumerate(loader):
        logits, feat, y, dbg = _forward(model, batch, device, args, want_debug=print_debug)
        loss = criterion(logits, y)
        losses.update(loss.item(), y.size(0))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
        if dbg:
            last_dbg = dbg
        if collect_outputs:
            logits_all.append(logits.detach().cpu())
            labels_all.append(y.detach().cpu())
            preds_all.append(pred.detach().cpu())
        if max_batches is not None and (bi + 1) >= max_batches:
            break

    acc = 100.0 * correct / max(1, total)
    if print_debug and last_dbg:
        print("[FirstLayer Debug]", last_dbg)

    if collect_outputs:
        outputs = dict(
            logits=torch.cat(logits_all) if logits_all else torch.empty((0,)),
            labels=torch.cat(labels_all) if labels_all else torch.empty((0,), dtype=torch.long),
            preds=torch.cat(preds_all) if preds_all else torch.empty((0,), dtype=torch.long),
        )
        return acc, losses.avg, outputs

    return acc, losses.avg


def _summarize_class_balance(labels, indices, tag):
    from collections import Counter

    cnt = Counter(int(labels[i]) for i in indices)
    total = float(len(indices))
    if total == 0:
        print(f"[{tag}] split为空")
        return

    values = np.array([cnt.get(k, 0) for k in sorted(cnt.keys())], dtype=np.float32)
    print(f"[{tag}] 总样本 {int(total)} | min={values.min()} max={values.max()} "
          f"mean={values.mean():.1f} std={values.std():.1f}")
    top_k = 10
    most_common = cnt.most_common(top_k)
    least_common = sorted(cnt.items(), key=lambda kv: kv[1])[:top_k]
    print(f"  Top{top_k} 类别(样本数): {most_common}")
    print(f"  Tail{top_k} 类别(样本数): {least_common}")


def _compute_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _classification_report_from_cm(cm):
    eps = 1e-9
    tp = np.diag(cm).astype(np.float64)
    per_class_sum = cm.sum(axis=1).astype(np.float64)
    pred_class_sum = cm.sum(axis=0).astype(np.float64)

    precision = tp / (pred_class_sum + eps)
    recall = tp / (per_class_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    support = per_class_sum
    accuracy = tp.sum() / max(cm.sum(), 1)
    macro_precision = float(np.nanmean(precision))
    macro_recall = float(np.nanmean(recall))
    macro_f1 = float(np.nanmean(f1))

    weighted_precision = float(np.nansum(precision * support) / max(support.sum(), eps))
    weighted_recall = float(np.nansum(recall * support) / max(support.sum(), eps))
    weighted_f1 = float(np.nansum(f1 * support) / max(support.sum(), eps))

    report = {
        "accuracy": float(accuracy * 100.0),
        "macro_precision": macro_precision * 100.0,
        "macro_recall": macro_recall * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "weighted_precision": weighted_precision * 100.0,
        "weighted_recall": weighted_recall * 100.0,
        "weighted_f1": weighted_f1 * 100.0,
        "per_class": [
            {
                "index": int(i),
                "precision": float(precision[i] * 100.0),
                "recall": float(recall[i] * 100.0),
                "f1": float(f1[i] * 100.0),
                "support": int(support[i]),
            }
            for i in range(len(tp))
        ],
    }
    return report


def _plot_confusion_matrix(cm, class_names, save_path, normalize=True, cmap='Blues'):
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=np.float64), where=cm_sum != 0)
    else:
        cm_display = cm.astype(np.float64)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_display, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.0 if cm_display.size else 0
    for i, j in np.ndindex(cm.shape):
        value = cm_display[i, j]
        if normalize:
            text = f"{value:.2f}"
        else:
            text = f"{int(value)}"
        plt.text(j, i, text,
                 horizontalalignment="center",
                 color="white" if value > thresh else "black",
                 fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_training_curves(history, save_path):
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    train_acc = [h.get('train_acc', float('nan')) for h in history]
    eval_acc = [h.get('val_acc', float('nan')) for h in history]
    train_loss = [h.get('train_loss', float('nan')) for h in history]
    val_loss = [h.get('val_loss', float('nan')) for h in history]
    train_eval_acc = [h.get('train_eval_acc', float('nan')) for h in history]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train (mixup)')
    plt.plot(epochs, eval_acc, label='Validation')
    plt.plot(epochs, train_eval_acc, label='Train (eval)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _save_report_csv(report, save_path):
    fieldnames = ['class_index', 'precision', 'recall', 'f1', 'support']
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in report['per_class']:
            writer.writerow({
                'class_index': item['index'],
                'precision': f"{item['precision']:.2f}",
                'recall': f"{item['recall']:.2f}",
                'f1': f"{item['f1']:.2f}",
                'support': item['support'],
            })


def _calc_topk_from_logits(logits, labels, ks=(1, 2, 3, 5)):
    if logits.numel() == 0:
        return {f"top{k}": float('nan') for k in ks}
    max_k = min(max(ks), logits.size(1))
    _, topk = torch.topk(logits, max_k, dim=1)
    results = {}
    for k in ks:
        kk = min(k, topk.size(1))
        corr = topk[:, :kk].eq(labels.view(-1, 1)).any(dim=1).float().mean().item() * 100.0
        results[f"top{k}"] = corr
    return results

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

    _summarize_class_balance(ys, tr_idx, "TRAIN")
    _summarize_class_balance(ys, va_idx, "VAL")
    _summarize_class_balance(ys, te_idx, "TEST")

    pin = device.type == 'cuda'
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)
    valloader   = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)
    testloader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=pin)

    # 额外取一部分训练集用于评估环节，帮助诊断训练/验证差距
    frac = float(getattr(args, 'train_eval_fraction', 0.25))
    frac = max(0.0, min(1.0, frac))
    if frac <= 0 or len(tr_idx) == 0:
        eval_indices = tr_idx
    elif frac >= 1.0:
        eval_indices = tr_idx
    else:
        n_eval = max(1, int(math.ceil(len(tr_idx) * frac)))
        eval_indices = rng.choice(tr_idx, size=n_eval, replace=False).tolist()
    train_eval_set = Subset(dataset, eval_indices)
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=pin,
    )

    # 看看 z/s 是否正常
    summarize_zs(trainloader, tag="train", max_batches=50)
    summarize_zs(valloader,  tag="val",   max_batches=50)

    # 模型（仅支持 P4 系列别名）
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
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
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
        a_tr_eval, l_tr_eval = evaluate(
            model, criterion, train_eval_loader, device, args,
            print_debug=False,
            max_batches=args.train_eval_max_batches,
        )
        a_ev, l_ev = evaluate(model, criterion, valloader, device, args, print_debug=args.first_debug)
        print(
            f"Train_Acc(%): {a_tr:.2f}  TrainEval_Acc(%): {a_tr_eval:.2f}  Eval_Acc(%): {a_ev:.2f}"
        )
        print(
            f"Train_Loss: {l_tr:.4f}  TrainEval_Loss: {l_tr_eval:.4f}  Eval_Loss: {l_ev:.4f}"
        )

        history.append({
            "epoch": int(epoch + 1),
            "train_acc": float(a_tr),
            "train_loss": float(l_tr),
            "train_eval_acc": float(a_tr_eval),
            "train_eval_loss": float(l_tr_eval),
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

    # 验证 & 测试
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    class_names = [str(i) for i in range(num_classes)]

    print("####### 训练集抽样复检")
    train_diag_acc, train_diag_loss, train_diag_outputs = evaluate(
        model, criterion, train_eval_loader, device, args,
        print_debug=False,
        max_batches=args.train_eval_max_batches,
        collect_outputs=True,
    )
    print(f"TrainEval_Acc(%): {train_diag_acc:.2f}  TrainEval_Loss: {train_diag_loss:.4f}")
    train_diag_topk = _calc_topk_from_logits(train_diag_outputs['logits'], train_diag_outputs['labels'])

    print("####### 验证集评估")
    acc_val, val_loss, val_outputs = evaluate(
        model, criterion, valloader, device, args,
        print_debug=True,
        collect_outputs=True,
    )
    print(f"Val_Acc(%): {acc_val:.2f}  Val_Loss: {val_loss:.4f}")

    print("####### 测试集评估")
    acc_test, test_loss, test_outputs = evaluate(
        model, criterion, testloader, device, args,
        print_debug=False,
        collect_outputs=True,
    )
    print(f"Test_Acc(%): {acc_test:.2f}  Test_Loss: {test_loss:.4f}")

    val_labels = val_outputs['labels'].numpy().astype(np.int64)
    val_preds = val_outputs['preds'].numpy().astype(np.int64)
    test_labels = test_outputs['labels'].numpy().astype(np.int64)
    test_preds = test_outputs['preds'].numpy().astype(np.int64)

    cm_val = _compute_confusion_matrix(val_labels, val_preds, num_classes)
    cm_test = _compute_confusion_matrix(test_labels, test_preds, num_classes)

    report_val = _classification_report_from_cm(cm_val)
    report_test = _classification_report_from_cm(cm_test)

    val_topk = _calc_topk_from_logits(val_outputs['logits'], val_outputs['labels'])
    test_topk = _calc_topk_from_logits(test_outputs['logits'], test_outputs['labels'])

    # 保存可视化和详细数据
    _plot_confusion_matrix(cm_val, class_names, osp.join(save_dir, 'val_confusion_matrix.png'), normalize=True)
    _plot_confusion_matrix(cm_val, class_names, osp.join(save_dir, 'val_confusion_matrix_counts.png'), normalize=False)
    _plot_confusion_matrix(cm_test, class_names, osp.join(save_dir, 'test_confusion_matrix.png'), normalize=True)
    _plot_confusion_matrix(cm_test, class_names, osp.join(save_dir, 'test_confusion_matrix_counts.png'), normalize=False)
    _plot_training_curves(history, osp.join(save_dir, 'training_curves.png'))
    _save_report_csv(report_val, osp.join(save_dir, 'val_classification_report.csv'))
    _save_report_csv(report_test, osp.join(save_dir, 'test_classification_report.csv'))

    np.savez_compressed(
        osp.join(save_dir, 'val_predictions.npz'),
        logits=val_outputs['logits'].numpy(),
        labels=val_labels,
        preds=val_preds,
    )
    np.savez_compressed(
        osp.join(save_dir, 'test_predictions.npz'),
        logits=test_outputs['logits'].numpy(),
        labels=test_labels,
        preds=test_preds,
    )
    np.savez_compressed(
        osp.join(save_dir, 'train_eval_predictions.npz'),
        logits=train_diag_outputs['logits'].numpy(),
        labels=train_diag_outputs['labels'].numpy().astype(np.int64),
        preds=train_diag_outputs['preds'].numpy().astype(np.int64),
    )

    metrics = {
        "val_acc": float(acc_val),
        "val_loss": float(val_loss),
        "val_macro_f1": float(report_val['macro_f1']),
        "val_macro_precision": float(report_val['macro_precision']),
        "val_macro_recall": float(report_val['macro_recall']),
        "val_top1": float(val_topk['top1']),
        "val_top3": float(val_topk.get('top3', float('nan'))),
        "val_top5": float(val_topk.get('top5', float('nan'))),
        "test_acc": float(acc_test),
        "test_loss": float(test_loss),
        "test_macro_f1": float(report_test['macro_f1']),
        "test_macro_precision": float(report_test['macro_precision']),
        "test_macro_recall": float(report_test['macro_recall']),
        "test_top1": float(test_topk['top1']),
        "test_top3": float(test_topk.get('top3', float('nan'))),
        "test_top5": float(test_topk.get('top5', float('nan'))),
        "train_eval_acc_final": float(train_diag_acc),
        "train_eval_loss_final": float(train_diag_loss),
        "train_eval_top1": float(train_diag_topk['top1']),
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
        "val_report": report_val,
        "test_report": report_test,
        "val_confusion_matrix": cm_val.tolist(),
        "test_confusion_matrix": cm_test.tolist(),
        "artifacts": {
            "training_curves": osp.join(save_dir, 'training_curves.png'),
            "val_confusion_matrix_norm": osp.join(save_dir, 'val_confusion_matrix.png'),
            "val_confusion_matrix_counts": osp.join(save_dir, 'val_confusion_matrix_counts.png'),
            "test_confusion_matrix_norm": osp.join(save_dir, 'test_confusion_matrix.png'),
            "test_confusion_matrix_counts": osp.join(save_dir, 'test_confusion_matrix_counts.png'),
            "val_report_csv": osp.join(save_dir, 'val_classification_report.csv'),
            "test_report_csv": osp.join(save_dir, 'test_classification_report.csv'),
            "val_predictions": osp.join(save_dir, 'val_predictions.npz'),
            "test_predictions": osp.join(save_dir, 'test_predictions.npz'),
            "train_eval_predictions": osp.join(save_dir, 'train_eval_predictions.npz'),
        },
    }

    json_path = osp.join(save_dir, f"{model_name}_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 训练日志已保存到: {json_path}")

    return summary


def main():
    """入口函数：直接在此修改数据路径和训练参数。"""

    config = SimpleNamespace(
        data='./data/ADS-B_0dB_train.mat',
        save_root='./runs_benchmark',
        class_num=100,
        batch_size=256,
        workers=0,
        lr=1e-4,
        wd=1e-4,
        max_epoch=100,
        patience=20,
        gpu='0',
        seed=42,
        first_debug=False,
        print_freq=5,
        shuffle_z=False,
        zero_z=False,
        zero_s=False,
        coef_scale=0.5,
        coef_gain=0.5,
        coef_cfo=0.5,
        coef_chirp=0.5,
        prior_drop=0.0,
        s_jitter=0.0,
        mixup_alpha=0.0,
        label_smoothing=0.05,
        train_eval_fraction=0.25,
        train_eval_max_batches=None,
        model_name='p4',
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    torch.manual_seed(config.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(config.seed)

    print("程序：P4 群卷积模型训练")
    print(f"数据路径：{config.data}")
    print(f"模型：{config.model_name}")
    print(
        "Steer系数: "
        f"scale={config.coef_scale} gain={config.coef_gain} "
        f"cfo={config.coef_cfo} chirp={config.coef_chirp}"
    )

    os.makedirs(config.save_root, exist_ok=True)

    save_dir = osp.join(config.save_root, config.model_name)
    run_one_model(config, config.model_name, config.data, save_dir, device)


if __name__ == '__main__':
    main()
