# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import re
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import warnings
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

# def check_anchor_order(m):
#     # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
#     a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
#     da = a[-1] - a[0]  # delta a
#     ds = m.stride[-1] - m.stride[0]  # delta s
#     if da and (da.sign() != ds.sign()):  # same order
#         print(f'{PREFIX}Reversing anchor order')
#         m.anchors[:] = m.anchors.flip(0)

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, 0.001, 0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x1, x2):
        return torch.cat((x1, x2), self.d)

class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc, anchors, stride, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.na = len(anchors) // 2  # number of anchors
        self.grid = torch.empty(0)  # init grid
        self.anchor_grid = torch.empty(0)  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(-1, 2)/stride)  # shape(nl,na,2)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = stride

    def forward(self, x):
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.dynamic or self.grid.shape[2:4] != x.shape[2:4]:
                self.grid, self.anchor_grid = self._make_grid(nx, ny)

            xy, wh, conf = x.sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid) * self.stride  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)

            return y.view(bs, self.na * nx * ny, self.no), x
        return x

    def _make_grid(self, nx=20, ny=20):
        d = self.anchors.device
        t = self.anchors.dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors * self.stride).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        print(f"{colorstr('activation:')} {act}")  # print
    na = 3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        print(m.__class__)
        if m in {
                Conv, Bottleneck, SPPF, C3, nn.ConvTranspose2d}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {C3}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class ModelForQuant(nn.Module):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.nc = self.yaml['nc']
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model[12] = nn.Identity()
        self.model[16] = nn.Identity()
        self.model[19] = nn.Identity()
        self.model[22] = nn.Identity()


    def forward(self, x):
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x4 = self.model[4](x)
        x = self.model[5](x4)
        x6 = self.model[6](x)
        x = self.model[7](x6)
        x = self.model[8](x)
        x = self.model[9](x)
        x10 = self.model[10](x)
        x = self.model[11](x10)
        x = torch.cat((x, x6), dim=1)
        x = self.model[13](x)
        x14 = self.model[14](x)
        x = self.model[15](x14)
        x = torch.cat((x, x4), dim=1)
        x17 = self.model[17](x)
        x = self.model[18](x17)
        x = torch.cat((x, x14), dim=1)
        x20 = self.model[20](x)
        x = self.model[21](x20)
        x = torch.cat((x, x10), dim=1)
        x23 = self.model[23](x)

        x24 = self.model[24](x17)
        x25 = self.model[25](x20)
        x26 = self.model[26](x23)
        return x24, x25, x26

class DetectionModel(nn.Module):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        self.model4quant=ModelForQuant(cfg, ch, nc, anchors )
        self.yaml = cfg  # model dict
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        self.detect_head1 = Detect(self.yaml["nc"], [10,13, 16,30, 33,23], 8)
        self.detect_head2 = Detect(self.yaml["nc"], [30,61, 62,45, 59,119],  16)
        self.detect_head3 = Detect(self.yaml["nc"], [116,90, 156,198, 373,326],  32)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model[12] = nn.Identity()
        self.model[16] = nn.Identity()
        self.model[19] = nn.Identity()
        self.model[22] = nn.Identity()

    # def _apply(self, fn):
    #     # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
    #     self = super()._apply(fn)
    #     m = self.model[-1]  # Detect()
    #     if isinstance(m, Detect):
    #         m.stride = fn(m.stride)
    #         m.grid = list(map(fn, m.grid))
    #         if isinstance(m.anchor_grid, list):
    #             m.anchor_grid = list(map(fn, m.anchor_grid))
    #     return self

    def forward(self, x):
        x1, x2, x3 = self.model4quant(x)
        y1, x1 = self.detect_head1(x1)
        y2, x2 = self.detect_head2(x2)
        y3, x3 = self.detect_head3(x3)
        y = torch.cat([y1, y2, y3], 1)
        
        return [y, [x1, x2, x3]]



def yolov5n(pretrain_path=None, model_path=None):
    cfg = {
        "nc": 80,
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "backbone": [
            [-1, 1, "Conv", [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
            [-1, 3, "C3", [128]],
            [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
            [-1, 6, "C3", [256]],
            [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
            [-1, 9, "C3", [512]],
            [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
            [-1, 3, "C3", [1024]],
            [-1, 1, "SPPF", [1024, 5]]  # 9
        ],
        "head":[
            [-1, 1, "Conv", [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
            [-1, 3, "C3", [512, False]],  # 13

            [-1, 1, "Conv", [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
            [-1, 3, "C3", [256, False]],  # 17 (P3/8-small)

            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat head P4
            [-1, 3, "C3", [512, False]],  # 20 (P4/16-medium)

            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],  # cat head P5
            [-1, 3, "C3", [1024, False]],  # 23 (P5/32-large)

            [17, 1, nn.Conv2d, [64, 255, 1]],  # Detect(P3, P4, P5)
            [20, 1, nn.Conv2d, [128, 255, 1]],  # Detect(P3, P4, P5)
            [23, 1, nn.Conv2d, [256, 255, 1]],  # Detect(P3, P4, P5)
        ]
    }
    model = DetectionModel(cfg)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {}
        for key, val in state_dict.items():
            key=key.replace("model.24.m.0", "model.24")
            key=key.replace("model.24.m.1", "model.25")
            key=key.replace("model.24.m.2", "model.26")
            if key!="model.24.anchors":
                new_state_dict[key] = val
        model.model4quant.load_state_dict(new_state_dict)
    return model

def yolov5s(pretrain_path=None, model_path=None):
    cfg = {
        "nc": 80,
        "depth_multiple": 0.33,
        "width_multiple": 0.50,
        "backbone": [
            [-1, 1, "Conv", [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
            [-1, 3, "C3", [128]],
            [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
            [-1, 6, "C3", [256]],
            [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
            [-1, 9, "C3", [512]],
            [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
            [-1, 3, "C3", [1024]],
            [-1, 1, "SPPF", [1024, 5]]  # 9
        ],
        "head":[
            [-1, 1, "Conv", [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
            [-1, 3, "C3", [512, False]],  # 13

            [-1, 1, "Conv", [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
            [-1, 3, "C3", [256, False]],  # 17 (P3/8-small)

            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat head P4
            [-1, 3, "C3", [512, False]],  # 20 (P4/16-medium)

            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],  # cat head P5
            [-1, 3, "C3", [1024, False]],  # 23 (P5/32-large)

            [17, 1, nn.Conv2d, [128, 255, 1]],  # Detect(P3, P4, P5)
            [20, 1, nn.Conv2d, [256, 255, 1]],  # Detect(P3, P4, P5)
            [23, 1, nn.Conv2d, [512, 255, 1]],  # Detect(P3, P4, P5)
        ]
    }
    model = DetectionModel(cfg)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {}
        for key, val in state_dict.items():
            key = "model."+key
            key=key.replace("model.24.m.0", "model.24")
            key=key.replace("model.24.m.1", "model.25")
            key=key.replace("model.24.m.2", "model.26")
            if key!="model.24.anchors":
                new_state_dict[key] = val
        model.model4quant.load_state_dict(new_state_dict)
    return model
