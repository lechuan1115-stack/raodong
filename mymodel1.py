#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Model definitions for the P4All and CNN-Transformer architectures."""

import torch.nn as nn

from gconv import SteerableFirstLayerPerSample


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), stride=(1, stride), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def _make_p4all_backbone():
    layers = [Conv2dBlock(32, 64, 11, pool=True)]
    layers += [Conv2dBlock(64, 128, 9, pool=True)]
    layers += [Conv2dBlock(128, 256, 7, pool=True)]
    layers += [Conv2dBlock(256, 256, 7, pool=True)]
    layers += [Conv2dBlock(256, 512, 5, pool=True)]
    layers += [Conv2dBlock(512, 512, 5, pool=True)]
    layers += [Conv2dBlock(512, 1024, 3, pool=True)]
    backbone = nn.Sequential(*layers)
    return backbone


class P4AllNet(nn.Module):
    """Steerable front-end that activates groups based on detected perturbations."""

    def __init__(self, n_classes=100, fs=50e6, debug_first=False,
                 coef_scale=1.0, coef_gain=1.0, coef_cfo=1.0, coef_chirp=1.0):
        super().__init__()
        self.first = SteerableFirstLayerPerSample(
            in_ch=2, out_ch=32, k=5, fs=fs,
            coef_scale=coef_scale, coef_gain=coef_gain,
            coef_cfo=coef_cfo, coef_chirp=coef_chirp,
        )
        self.features = _make_p4all_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )
        self.set_first_debug(debug_first)

    def set_steer_disabled(self, flag: bool):
        self.first.set_disable_steer(flag)

    def set_first_debug(self, flag: bool):
        self.first.set_debug(flag)

    def get_first_last_debug(self):
        return getattr(self.first, "last_debug", {})

    def forward(self, x, z=None, s=None):
        x = self.first(x, z, s) if (z is not None and s is not None) else self.first(x, None, None)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        feat = x
        logits = self.classifier(x)
        return logits, feat


class CNNTransformer(nn.Module):
    """Hybrid CNN + Transformer baseline for perturbation-agnostic inference."""

    def __init__(self, n_classes=100, embed_dim=256, depth=4, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, n_classes),
        )

    def set_steer_disabled(self, flag: bool):
        # Interface compatibility with training utilities
        return

    def set_first_debug(self, flag: bool):
        return

    def get_first_last_debug(self):
        return {}

    def forward(self, x, *args, **kwargs):
        x = x.squeeze(2)  # [B,2,L]
        feats = self.conv(x)
        tokens = feats.transpose(1, 2)  # [B,L',C]
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        feat = pooled
        logits = self.head(pooled)
        return logits, feat


_FACTORY = {
    'p4all': P4AllNet,
    'cnn-transformer': CNNTransformer,
}


def create(name, num_classes, fs=50e6, debug_first=False, **kwargs):
    key = name.lower()
    if key not in _FACTORY:
        raise KeyError(f"Unknown model: {name}")
    if key == 'p4all':
        return _FACTORY[key](
            n_classes=num_classes,
            fs=fs,
            debug_first=debug_first,
            **kwargs,
        )
    return _FACTORY[key](n_classes=num_classes, **kwargs)
# -*- coding:utf-8 -*-import torchimport torch.nn as nnimport torch.nn.functional as Ffrom gconv import PlainFirstLayer, SteerableFirstLayerPerSample# --------------------------------# 共享：堆叠主干（保持与原 Steer 版一致）# --------------------------------def _make_backbone():    def C(cin, cout, k):        return nn.Conv2d(cin, cout, kernel_size=(1,k), padding=(0,k//2), bias=True)    class BlurPool1D(nn.Module):        def __init__(self, channels, stride=2):            super().__init__()            self.pool = nn.AvgPool2d(kernel_size=(1, stride), stride=(1, stride))        def forward(self, x):            return self.pool(x)    blocks = []    blocks += [nn.ReLU(), nn.BatchNorm2d(32)]    blocks += [BlurPool1D(32, stride=2)]    blocks += [C(32,128,11), nn.ReLU(), nn.BatchNorm2d(128),               C(128,128,11), nn.ReLU(), nn.BatchNorm2d(128),               BlurPool1D(128, stride=2)]    blocks += [C(128,256,11), nn.ReLU(), nn.BatchNorm2d(256),               C(256,256,7),  nn.ReLU(), nn.BatchNorm2d(256),               BlurPool1D(256, stride=2)]    blocks += [C(256,256,7), nn.ReLU(), nn.BatchNorm2d(256),               C(256,256,7), nn.ReLU(), nn.BatchNorm2d(256),               BlurPool1D(256, stride=2)]    blocks += [C(256,256,7), nn.ReLU(), nn.BatchNorm2d(256),               C(256,256,7), nn.ReLU(), nn.BatchNorm2d(256),               BlurPool1D(256, stride=2)]    blocks += [C(256,512,7), nn.ReLU(), nn.BatchNorm2d(512),               C(512,512,5), nn.ReLU(), nn.BatchNorm2d(512),               BlurPool1D(512, stride=2)]    blocks += [C(512,512,5), nn.ReLU(), nn.BatchNorm2d(512),               C(512,512,5), nn.ReLU(), nn.BatchNorm2d(512),               BlurPool1D(512, stride=2)]    blocks += [C(512,1024,3), nn.ReLU(),               C(1024,1024,3), nn.ReLU(), nn.BatchNorm2d(1024),               BlurPool1D(1024, stride=2)]    blocks += [C(1024,1024,3), nn.ReLU(), nn.BatchNorm2d(1024),               C(1024,1024,3), nn.ReLU(), nn.BatchNorm2d(1024),               nn.AdaptiveAvgPool2d(1)]    return nn.Sequential(*blocks)def _make_classifier(n_classes):    return nn.Sequential(        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),        nn.Linear(512, n_classes)    )# --------------------------------# 有群版本：第一层为可转向# --------------------------------class PerturbAwareNetSteer(nn.Module):    def __init__(self, n_classes=100, fs=50e6, use_blurpool=True,                 debug_first=False,                 coef_scale=1.0, coef_gain=1.0, coef_cfo=1.0, coef_chirp=1.0):        super().__init__()        self.steer = SteerableFirstLayerPerSample(            in_ch=2, out_ch=32, k=5, fs=fs,            coef_scale=coef_scale, coef_gain=coef_gain, coef_cfo=coef_cfo, coef_chirp=coef_chirp        )        self.steer.set_debug(debug_first)        self.features   = _make_backbone()        self.classifier = _make_classifier(n_classes)    # 控制接口（csr.py 会调用）    def set_steer_disabled(self, flag: bool):        self.steer.set_disable_steer(flag)    def set_first_debug(self, flag: bool):        self.steer.set_debug(flag)    def get_first_last_debug(self):        return getattr(self.steer, "last_debug", {})    def forward(self, x, z=None, s=None):        if (z is not None) and (s is not None):            x = self.steer(x, z, s)        else:            x = self.steer(x, None, None)  # 等价于普通卷积        x = self.features(x)        x = x.view(x.size(0), -1)        feat = x        logits = self.classifier(x)        return logits, feat# --------------------------------# 真·基线：第一层为 PlainFirstLayer（不接受 z/s，不做转向）# --------------------------------class PlainCNNBaseline(nn.Module):    def __init__(self, n_classes=100, fs=50e6, use_blurpool=True, **kwargs):        super().__init__()        self.first      = PlainFirstLayer(in_ch=2, out_ch=32, k=5)        self.features   = _make_backbone()        self.classifier = _make_classifier(n_classes)    # 用于与有群版统一接口，但这里不做任何事    def set_steer_disabled(self, flag: bool):        return    def set_first_debug(self, flag: bool):        return    def get_first_last_debug(self):        return {}    def forward(self, x, *args, **kwargs):        x = self.first(x)             # 忽略 z/s        x = self.features(x)        x = x.view(x.size(0), -1)        feat = x        logits = self.classifier(x)        return logits, feat# 工厂__factory = {    'PerturbAwareNetSteer': PerturbAwareNetSteer,  # 有群（可转向）    'PerturbAwareNet':      PerturbAwareNetSteer,    'PerturbAblationNet':   PlainCNNBaseline,      # ✅ 改为真·基线}def create(name, num_classes, fs=50e6, use_blurpool=True, debug_first=False, **kwargs):    if name not in __factory:        raise KeyError(f"Unknown model: {name}")    cls = __factory[name]    # baseline 不需要 debug_first；有群模型保留    if name == 'PerturbAblationNet':        return cls(n_classes=num_classes, fs=fs, use_blurpool=use_blurpool)    return cls(n_classes=num_classes, fs=fs, use_blurpool=use_blurpool, debug_first=debug_first, **kwargs)