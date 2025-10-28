#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

PERTURB_ORDER = ['CFO', 'SCALE', 'GAIN', 'SHIFT', 'CHIRP']

def _grid(minmax, steps):
    lo, hi = float(minmax[0]), float(minmax[1])
    if steps > 1:
        return torch.linspace(lo, hi, steps, dtype=torch.float32)
    return torch.tensor([(lo+hi)/2], dtype=torch.float32)

class PlainFirstLayer(nn.Module):
    """普通卷积第一层（无群；消融用）"""
    def __init__(self, in_ch=2, out_ch=32, k=5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), padding=(0, k//2), bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, a=0.2, nonlinearity='relu')
    def forward(self, x): return self.conv(x)

# ---------- 群作用：核级逐样本 steering ----------
def steer_scale_filters_per_sample(w, alpha):
    # w:[B,OC,IC,1,kW], alpha:[B]
    B, OC, IC, _, kW = w.shape
    ker = w.view(B*OC*IC, 1, kW)
    out_list=[]; start=0
    for b in range(B):
        ab = torch.clamp(alpha[b], min=1e-3)
        new_W = int(torch.clamp((kW / ab).round(), min=1).item())
        seg = ker[start:start+OC*IC]
        seg2 = F.interpolate(seg, size=new_W, mode='linear', align_corners=False)
        seg3 = F.interpolate(seg2, size=kW,   mode='linear', align_corners=False)
        out_list.append(seg3)
        start += OC*IC
    ker2 = torch.cat(out_list, dim=0)
    return ker2.view(B,OC,IC,1,kW)

def _mix_IQ_by_phase_per_sample(w, phi):
    # w:[B,OC,IC,1,kW], phi:[B,kW]
    B, OC, IC, _, kW = w.shape
    if IC < 2: return w
    I = w[:,:,0,0,:]; Q = w[:,:,1,0,:]
    c = torch.cos(phi).unsqueeze(1); s = torch.sin(phi).unsqueeze(1)
    I2 = I*c - Q*s; Q2 = I*s + Q*c
    w2 = w.clone(); w2[:,:,0,0,:]=I2; w2[:,:,1,0,:]=Q2
    return w2

def steer_cfo_filters_per_sample(w, fs, f0):
    kW = w.shape[-1]
    n  = torch.arange(kW, device=w.device, dtype=torch.float32)[None,:]
    phi = 2.0*torch.pi * f0[:,None] * n / fs
    return _mix_IQ_by_phase_per_sample(w, phi)

def steer_chirp_filters_per_sample(w, fs, a):
    kW = w.shape[-1]
    n  = torch.arange(kW, device=w.device, dtype=torch.float32)[None,:]
    t  = n / fs
    phi = torch.pi * a[:,None] * (t**2)
    return _mix_IQ_by_phase_per_sample(w, phi)

def steer_gain_filters_per_sample(w, rho, theta):
    # rho:[B], theta:[B]
    B, OC, IC, _, _ = w.shape
    if IC >= 2:
        I = w[:,:,0,0,:]; Q = w[:,:,1,0,:]
        c = torch.cos(theta)[:,None,None]; s = torch.sin(theta)[:,None,None]
        I2 = c*I - s*Q; Q2 = s*I + c*Q
        w = w.clone(); w[:,:,0,0,:]=I2; w[:,:,1,0,:]=Q2
    rho = rho[:,None,None,None,None]
    return rho * w

def steer_shift_filters_per_sample(w, shift):
    """Apply circular time shift to the kernels on a per-sample basis."""
    B, _, _, _, _ = w.shape
    shift = shift.to(torch.int64)
    if torch.all(shift == 0):
        return w
    w_out = w.clone()
    for b in range(B):
        k = int(shift[b].item())
        if k != 0:
            w_out[b] = torch.roll(w_out[b], shifts=k, dims=-1)
    return w_out

class SteerableFirstLayerPerSample(nn.Module):
    """
    逐样本群等变第一层：基核 + 核转向 + grouped conv
    顺序：SCALE -> CFO -> CHIRP -> GAIN -> SHIFT，逐项检测对应扰动后再启用。
    新增：coef_* 参数可调扰动强度
    """
    def __init__(self, in_ch=2, out_ch=32, k=5,
                 fs=50e6,
                 gain_amp_range=(0.7,1.3),
                 gain_phase_range=(0.0,0.0),
                 scale_range=(0.98,1.02),
                 cfo_range=(-800.0,800.0),
                 chirp_range=(-2e3,2e3),
                 # 新增：扰动强度系数
                 coef_scale=1.0, coef_gain=1.0, coef_cfo=1.0, coef_chirp=1.0,
                 ):
        super().__init__()
        self.fs = float(fs)
        self.base = nn.Conv2d(in_ch, out_ch, kernel_size=(1,k), padding=(0,k//2), bias=False)
        nn.init.kaiming_uniform_(self.base.weight, a=0.2, nonlinearity='relu')

        # 强制关闭 & 调试
        self.force_disable = False
        self.debug = False
        self.last_debug = {}

        # 系数
        self.coef_scale  = float(coef_scale)
        self.coef_gain   = float(coef_gain)
        self.coef_cfo    = float(coef_cfo)
        self.coef_chirp  = float(coef_chirp)

        # 可选网格（占位）
        self.register_buffer('grid_gain',  _grid(gain_amp_range,   9))
        self.register_buffer('grid_th',    _grid(gain_phase_range, 1))
        self.register_buffer('grid_scale', _grid(scale_range,      9))
        self.register_buffer('grid_cfo',   _grid(cfo_range,       11))
        self.register_buffer('grid_chirp', _grid(chirp_range,     11))

    # 外部控制
    def set_disable_steer(self, flag: bool):
        self.force_disable = bool(flag)
    def set_debug(self, flag: bool):
        self.debug = bool(flag)

    def forward(self, x, z=None, s=None):
        # 关闭群转向
        if self.force_disable or z is None or s is None:
            if self.debug: print(json.dumps({"steer_applied": False}))
            return self.base(x)

        B, _, _, L = x.shape
        w0 = self.base.weight                    # [OC,IC,1,kW]
        wB = w0.unsqueeze(0).repeat(B,1,1,1,1)   # [B,OC,IC,1,kW]
        fs = torch.tensor(self.fs, dtype=torch.float32, device=x.device)

        applied = {tag: False for tag in PERTURB_ORDER}

        def _apply_group(tag, neutral, transform_fn):
            idx = PERTURB_ORDER.index(tag)
            active = (z[:, idx] > 0.5)
            if not torch.any(active):
                return
            param = torch.nan_to_num(s[:, idx])
            param = transform_fn(param)
            full_param = torch.full_like(param, neutral)
            full_param[active] = param[active]
            nonlocal wB
            if tag == 'CFO':
                wB = steer_cfo_filters_per_sample(wB, fs, full_param)
            elif tag == 'SCALE':
                wB = steer_scale_filters_per_sample(wB, full_param)
            elif tag == 'GAIN':
                theta = torch.zeros_like(full_param)
                wB = steer_gain_filters_per_sample(wB, full_param, theta)
            elif tag == 'CHIRP':
                wB = steer_chirp_filters_per_sample(wB, fs, full_param)
            elif tag == 'SHIFT':
                wB = steer_shift_filters_per_sample(wB, full_param.round())
            applied[tag] = bool(active.any())

        _apply_group('SCALE', 1.0, lambda v: 1.0 + self.coef_scale * (v - 1.0))
        _apply_group('CFO',   0.0, lambda v: self.coef_cfo * v)
        _apply_group('CHIRP', 0.0, lambda v: self.coef_chirp * v)
        _apply_group('GAIN',  1.0, lambda v: 1.0 + self.coef_gain * (v - 1.0))
        _apply_group('SHIFT', 0.0, lambda v: v)

        OC, IC, _, kW = w0.shape
        xG = x.reshape(1, B*IC, 1, L)               # [1,B*IC,1,L]
        wG = wB.reshape(B*OC, IC, 1, kW)            # [B*OC,IC,1,kW]
        yG = F.conv2d(xG, wG, bias=None, stride=1, padding=(0,kW//2), groups=B)
        y  = yG.view(B, OC, 1, L)

        if self.debug:
            with torch.no_grad():
                base = w0.detach().flatten(start_dim=0).float()
                cur  = wB.detach().mean(dim=0).flatten(start_dim=0).float()
                diff = torch.norm(cur - base) / (torch.norm(base) + 1e-8)
                self.last_debug = {
                    "steer_applied": True,
                    "deltaW_rel_norm_mean": float(diff.item()),
                    **{tag: bool(applied[tag]) for tag in applied},
                }
        return y
