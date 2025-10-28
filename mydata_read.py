#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import h5py, os, json, os.path as osp

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover - scipy 可能缺失
    loadmat = None

# 与 gconv 中顺序保持一致
PERTURB_TAGS = ['CFO', 'SCALE', 'GAIN', 'SHIFT', 'CHIRP']

def _check_path(p):
    full = os.path.abspath(p)
    print(f"[DATA] trying to read: {full}")
    if not os.path.exists(full):
        raise FileNotFoundError(full)
    return full

def _as_complex_NL(A):
    A = np.array(A)
    if np.iscomplexobj(A):
        X = A if A.ndim == 2 else A.reshape(A.shape[0], -1)
        return X.astype(np.complex64)

    # 找到 size==2 的轴作为IQ通道轴
    axis = [i for i, s in enumerate(A.shape) if s == 2]
    if axis:
        ch = axis[0]
        Am = np.moveaxis(A, ch, 1)  # (...,2,...) -> axis=1
        prefix = Am.shape[0]
        suffix = int(np.prod(Am.shape[2:])) if Am.ndim > 2 else 1
        A3 = Am.reshape(prefix, 2, suffix)  # [p,2,s]
        if suffix >= prefix:
            # [N,L,2] 情况
            Ap = np.transpose(A3, (2, 0, 1))  # [N,L,2]
            return (Ap[:, :, 0] + 1j * Ap[:, :, 1]).astype(np.complex64)
        else:
            # [L,2,s] 情况
            return (A3[:, 0, :] + 1j * A3[:, 1, :]).astype(np.complex64)

    # 常见几种排布
    if A.ndim == 3 and A.shape[1] == 2:
        return (A[:, 0, :] + 1j * A[:, 1, :]).astype(np.complex64)
    if A.ndim == 3 and A.shape[2] == 2:
        return (A[:, :, 0] + 1j * A[:, :, 1]).astype(np.complex64)
    if A.ndim == 2 and A.shape[0] == 2:
        return (A[0, :] + 1j * A[1, :])[None, :].astype(np.complex64)
    if A.ndim == 2 and A.shape[1] == 2:
        return (A[:, 0] + 1j * A[:, 1])[None, :].astype(np.complex64)
    if A.ndim == 1:
        return A.reshape(1, -1).astype(np.complex64)

    raise RuntimeError(f"未识别 I/Q 格式, shape={A.shape}")

def _complex_to_P4_input(X):
    I = np.real(X).astype(np.float32)
    Q = np.imag(X).astype(np.float32)
    X4 = np.stack([I, Q], axis=1)   # [N,2,L]
    return X4[:, :, None, :]        # [N,2,1,L]

def _maybe_read_S(hf, N):
    """
    尝试读取扰动参数 S，返回形状[N,5]或None。
    优先使用键 'S'（支持 (N,5) 或 (5,N) 自动转置）。
    如果没有 S，再尝试从 'disturb_params' 解析（常见 -v7.3 cell/struct）。
    """
    # 1) 直接使用 'S'
    if 'S' in hf.keys():
        try:
            S_raw = np.array(hf['S'])
            # 自动转置 (5,N) -> (N,5)
            if S_raw.ndim == 2:
                if S_raw.shape == (N, 5):
                    S = S_raw.astype(np.float32)
                    print("[INFO] S loaded as (N,5)")
                    return S
                if S_raw.shape == (5, N):
                    S = S_raw.T.astype(np.float32)
                    print("[INFO] S transposed from (5,N) -> (N,5)")
                    return S
                # 其他情况，尝试靠维度自动修正
                if S_raw.shape[0] == 5:
                    S = S_raw.T.astype(np.float32)
                    print("[INFO] S transposed by heuristics (first dim=5)")
                    return S
                if S_raw.shape[1] == 5:
                    S = S_raw.astype(np.float32)
                    print("[INFO] S accepted by heuristics (second dim=5)")
                    return S
            print(f"[WARN] S shape mismatch: {S_raw.shape} -> ignoring S")
        except Exception as e:
            print("[WARN] reading S failed:", e)

    # 2) 尝试从 'disturb_params' 解析
    if 'disturb_params' in hf.keys():
        try:
            dp = hf['disturb_params']
            arr = np.array(dp, dtype=object)
            if arr.ndim == 2:
                arr = arr.reshape(-1)  # (1,N)->(N,)
            cols = {k: np.full((N,), np.nan, np.float32) for k in PERTURB_TAGS}
            for i in range(min(N, arr.size)):
                item = arr[i]
                # -v7.3 的结构体在 Python 侧经常不是 dict，实际解析较复杂；
                # 这里保留占位，若以后需要，可按实际结构扩展。
                # 当前数据已经有 S，所以通常不会走到这里。
                _ = item  # 占位
            S = np.stack([cols[k] for k in PERTURB_TAGS], axis=1).astype(np.float32)
            print("[INFO] fallback S built from disturb_params (likely NaNs)")
            return S
        except Exception as e:
            print("[WARN] disturb_params parse failed:", e)

    return None

def _is_mat73(mat_path):
    try:
        with open(mat_path, 'rb') as fh:
            header = fh.read(128)
        return b'MATLAB 7.3 MAT-file' in header
    except OSError:
        return False


def _load_mat_dict(mat_path):
    if _is_mat73(mat_path):
        print("[INFO] detected MATLAB v7.3 MAT-file, routing to HDF5 loader")
        return None
    if loadmat is None:
        raise ImportError("scipy.io.loadmat 未安装，无法读取 .mat 数据")
    kwargs = dict(struct_as_record=False, squeeze_me=True)
    try:
        mdict = loadmat(mat_path, **kwargs, simplify_cells=True)
    except TypeError:
        mdict = loadmat(mat_path, **kwargs)
    return {k: v for k, v in mdict.items() if not k.startswith('__')}


def _mat_coerce_numeric(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            flat = [
                _mat_coerce_numeric(item)
                for item in obj.reshape(-1)
            ]
            if len(flat) == 0:
                return np.array([])
            if all(isinstance(v, np.ndarray) and v.shape == flat[0].shape for v in flat):
                return np.stack(flat, axis=0)
            try:
                return np.array(flat)
            except Exception:
                return np.array(flat, dtype=object)
        return obj
    if np.isscalar(obj):
        return np.array(obj)
    if hasattr(obj, '_fieldnames'):
        d = {name: _mat_coerce_numeric(getattr(obj, name)) for name in obj._fieldnames}
        # 常见 real/imag 拆分
        if {'real', 'imag'}.issubset(d.keys()):
            return d['real'] + 1j * d['imag']
        return d
    if isinstance(obj, (list, tuple)):
        flat = [_mat_coerce_numeric(v) for v in obj]
        try:
            return np.stack(flat, axis=0)
        except Exception:
            return np.array(flat, dtype=object)
    try:
        return np.array(obj)
    except Exception:
        return None


def _mat_pick(mdict, candidates):
    if mdict is None:
        return None
    for key in candidates:
        if key in mdict:
            return _mat_coerce_numeric(mdict[key])
    return None


def load_h5_with_perturb(h5_path, for_p4=True, shuffle=True, seed=42):
    h5_path = _check_path(h5_path)
    with h5py.File(h5_path, 'r') as hf:
        keys = list(hf.keys())
        print(json.dumps({"keys": list(keys)}, ensure_ascii=False))

        # ---- IQ 数据字段 ----
        iq_key = None
        for k in ['train', 'data', 'IQ', 'iq', 'iq_data', 'signal', 'x', 'data_iq', 'test']:
            if k in hf.keys():
                iq_key = k
                break
        if iq_key is None:
            # 回退：第一个 dataset
            for k in keys:
                if isinstance(hf[k], h5py.Dataset):
                    iq_key = k
                    break
        if iq_key is None:
            raise RuntimeError("无法找到 I/Q 字段")
        A = np.array(hf[iq_key])

        # ---- Y 标签（可选）----
        Y = None
        for k in ['trainlabel', 'testlabel', 'label', 'y', 'labels']:
            if k in hf.keys():
                try:
                    Y = np.array(hf[k]).squeeze()
                except Exception:
                    pass
                break

        # ---- 采样率 fs ----
        fs = 50e6
        for k in ['fs', 'Fs', 'FS', 'sample_rate', 'samp_rate', 'sampling_rate', 'sampling_frequency']:
            if k in hf.keys():
                try:
                    fs = float(np.array(hf[k]).squeeze())
                    break
                except Exception:
                    pass

        # ---- Z （是否激活的 one-hot/0-1）----
        Z = np.array(hf['Z']).astype(np.float32) if 'Z' in hf.keys() else None

        # ---- S （扰动参数）----
        N_guess = A.shape[0] if A is not None else (Z.shape[0] if Z is not None else None)
        S = _maybe_read_S(hf, N_guess) if N_guess is not None else None

    # ---- 统一为 [N,L] 复数数组，并转换成 P4 输入 ----
    Xc = _as_complex_NL(A)
    if Xc.ndim == 1:
        Xc = Xc[None, :]
    N, L = Xc.shape

    X = _complex_to_P4_input(Xc) if for_p4 else np.stack(
        [np.real(Xc).astype(np.float32), np.imag(Xc).astype(np.float32)], axis=1
    )

    # ---- Y 对齐 ----
    if Y is not None:
        Y = np.array(Y).squeeze().astype(np.float32)
        if Y.shape == ():
            Y = np.repeat(Y, N)
    else:
        Y = np.zeros((N,), dtype=np.float32)

    # ---- Z/S 对齐到 N ----
    def _fix2(a, name):
        if a is None:
            return None
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[0] == N:
            return a.astype(np.float32)
        if a.ndim == 2 and a.shape[1] == N:
            return a.T.astype(np.float32)
        print(f"[WARN] {name} shape mismatch -> ignoring {name}")
        return None

    Z = _fix2(Z, 'Z')
    S = _fix2(S, 'S')

    # ---- 可选 shuffle 保持配对 ----
    idx = np.arange(N)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    if Z is not None:
        Z = Z[idx]
    if S is not None:
        S = S[idx]

    print(json.dumps({
        "summary": f"X{X.shape}, Y{Y.shape}, Z{None if Z is None else Z.shape}, "
                   f"S{None if S is None else S.shape}, fs={fs}"
    }, ensure_ascii=False))
    return X.astype(np.float32), Y.astype(np.float32), Z, S, float(fs)


def load_mat_with_perturb(mat_path, for_p4=True, shuffle=True, seed=42):
    mat_path = _check_path(mat_path)
    mdict = _load_mat_dict(mat_path)
    if mdict is None:
        # MATLAB v7.3 -> 直接当做 HDF5 读取
        return load_h5_with_perturb(mat_path, for_p4=for_p4, shuffle=shuffle, seed=seed)
    print(json.dumps({"keys": sorted(mdict.keys())}, ensure_ascii=False))

    A = _mat_pick(mdict, ['train', 'data', 'IQ', 'iq', 'iq_data', 'signal', 'x', 'data_iq', 'test'])
    if isinstance(A, dict):
        raise RuntimeError("无法从 .mat 结构中解析 I/Q 数据，请检查字段命名")
    if A is None:
        raise RuntimeError("无法在 .mat 文件中找到 I/Q 数据字段")

    Y = _mat_pick(mdict, ['trainlabel', 'testlabel', 'label', 'y', 'labels'])
    if isinstance(Y, dict):
        Y = None

    fs_raw = _mat_pick(mdict, ['fs', 'Fs', 'FS', 'sample_rate', 'samp_rate', 'sampling_rate', 'sampling_frequency'])
    try:
        fs = float(np.array(fs_raw).squeeze()) if fs_raw is not None else 50e6
    except Exception:
        fs = 50e6

    Z = _mat_pick(mdict, ['Z'])
    if isinstance(Z, dict):
        Z = None

    S = _mat_pick(mdict, ['S', 'disturb_params'])
    if isinstance(S, dict):
        S = None

    Xc = _as_complex_NL(A)
    if Xc.ndim == 1:
        Xc = Xc[None, :]
    N, _ = Xc.shape

    X = _complex_to_P4_input(Xc) if for_p4 else np.stack(
        [np.real(Xc).astype(np.float32), np.imag(Xc).astype(np.float32)], axis=1
    )

    if Y is not None:
        Y = np.array(Y).squeeze().astype(np.float32)
        if Y.shape == ():
            Y = np.repeat(Y, N)
    else:
        Y = np.zeros((N,), dtype=np.float32)

    def _fix2(a, name):
        if a is None:
            return None
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[0] == N:
            return a.astype(np.float32)
        if a.ndim == 2 and a.shape[1] == N:
            return a.T.astype(np.float32)
        print(f"[WARN] {name} shape mismatch -> ignoring {name}")
        return None

    Z = _fix2(Z, 'Z')
    S = _fix2(S, 'S')

    idx = np.arange(N)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    if Z is not None:
        Z = Z[idx]
    if S is not None:
        S = S[idx]

    print(json.dumps({
        "summary": f"X{X.shape}, Y{Y.shape}, Z{None if Z is None else Z.shape}, "
                   f"S{None if S is None else S.shape}, fs={fs}"
    }, ensure_ascii=False))
    return X.astype(np.float32), Y.astype(np.float32), Z, S, float(fs)


def load_data_with_perturb(path, for_p4=True, shuffle=True, seed=42):
    ext = osp.splitext(path)[1].lower()
    if ext == '.mat':
        return load_mat_with_perturb(path, for_p4=for_p4, shuffle=shuffle, seed=seed)
    return load_h5_with_perturb(path, for_p4=for_p4, shuffle=shuffle, seed=seed)

class SignalDatasetWithPerturb(Dataset):
    def __init__(self, data_path, for_p4=True, transform=None, shuffle=True, seed=42):
        X, Y, Z, S, fs = load_data_with_perturb(data_path, for_p4=for_p4, shuffle=shuffle, seed=seed)
        self.X, self.Y, self.Z, self.S, self.fs = X, Y, Z, S, fs
        self.transform = transform

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.Z is None:
            return x, y
        z = self.Z[i]
        if self.S is not None:
            s = self.S[i]
        else:
            s = np.full((len(PERTURB_TAGS),), np.nan, np.float32)
        return x, y, z, s

    def __len__(self):
        return len(self.X)

class SignalDataset1(SignalDatasetWithPerturb):
    def __init__(self, p, transform=None, shuffle=True, seed=42):
        super().__init__(p, for_p4=True, transform=transform, shuffle=shuffle, seed=seed)
