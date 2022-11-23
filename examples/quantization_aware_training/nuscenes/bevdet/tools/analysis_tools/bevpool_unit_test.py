# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module

from mmdet3d.ops import bev_pool


#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#        self.register_buffer("kept", torch.from_numpy(np.load("/data/githubs/BEVDet/constant_params/kept.npy")))
#        self.register_buffer("indices", torch.from_numpy(np.load("/data/githubs/BEVDet/constant_params/indices.npy")))
#        self.register_buffer("coords", torch.from_numpy(np.load("/data/githubs/BEVDet/constant_params/coords.npy")))
#        self.register_buffer("interval_starts", torch.from_numpy(np.load("/data/githubs/BEVDet/constant_params/interval_starts.npy")))
#        self.register_buffer("interval_lengths", torch.from_numpy(np.load("/data/githubs/BEVDet/constant_params/interval_lengths.npy")))
#        self.nx = torch.tensor([128, 128, 1], dtype=torch.long)
#
#    def forward(self, x):
#        y = bev_pool(x, self.coords, self.interval_lengths, self.interval_starts, 1, self.nx[2], self.nx[0], self.nx[1])
#        return y
#
#
#def main():
#    input_tensor = torch.from_numpy(np.load("/data/githubs/BEVDet/input.npy")).cuda()
#
#    model = Model().cuda()
#    model.eval()
#    out = model(input_tensor)
#    from IPython import embed; embed()
#    with torch.no_grad():
#        torch.onnx.export(
#            model,
#            input_tensor,
#            "bevpool.onnx",
#            export_params=True,
#            opset_version=11,
#        )

from mmdet.models import DETECTORS
from mmdet3d.models import BEVDet

@DETECTORS.register_module()
class BEVDetONNX(BEVDet):
    def onnx_export(self, img_inputs, img_metas=None):
        x = self.image_encoder(img_inputs[0])
        x = self.img_view_transformer([x] + img_inputs[1:])
        x = [self.bev_encoder(x)]
        outs = self.pts_bbox_head(x)
        return outs

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, img_inputs, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            img_inputs_cuda = [im.cuda() for im in img_inputs]
            return self.onnx_export(img_inputs_cuda)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)




if __name__ == '__main__':
    main()
