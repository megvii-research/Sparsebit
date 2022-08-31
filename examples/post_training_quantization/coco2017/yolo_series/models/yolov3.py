from collections import OrderedDict

import torch
import torch.nn as nn


def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1) -> nn.Sequential:
    "Create a seuence Conv2d->BatchNorm2d->ReLu layer."
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        ni,
                        nf,
                        kernel_size=ks,
                        bias=False,
                        stride=stride,
                        padding=ks // 2,
                    ),
                ),
                ("bn", nn.BatchNorm2d(nf)),
                ("relu", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ]
        )
    )


class ResLayer(nn.Module):
    "Resnet style layer with `ni` inputs."

    def __init__(self, ni: int):
        super(ResLayer, self).__init__()
        self.layer1 = conv_bn_lrelu(ni, ni // 2, ks=1)
        self.layer2 = conv_bn_lrelu(ni // 2, ni, ks=3)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Darknet(nn.Module):
    def make_group_layer(self, ch_in: int, num_blocks: int, stride: int = 1):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        return [conv_bn_lrelu(ch_in, ch_in * 2, stride=stride)] + [
            (ResLayer(ch_in * 2)) for i in range(num_blocks)
        ]

    def __init__(self, depth=53, ch_in=3, nf=32):
        """
        depth (int): depth of darknet used in model, usually use [21, 53] for this param
        ch_in (int): input channels, for example, ch_in of RGB image is 3
        nf (int): number of filters output in stem.
        out_features (List[str]): desired output layer name.
        num_classes (int): For ImageNet, num_classes is 1000. If None, no linear layer will be
            added.
        """
        super(Darknet, self).__init__()
        self.stem = conv_bn_lrelu(ch_in, nf, ks=3, stride=1)

        current_stride = 1

        "create darknet with `nf` and `num_blocks` layers"
        self.stages_and_names = []
        num_blocks = [1, 2, 8, 8, 4]

        for i, nb in enumerate(num_blocks):
            stage = nn.Sequential(*self.make_group_layer(nf, nb, stride=2))
            name = "dark" + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            current_stride *= 2
            nf *= 2

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        for stage, _ in self.stages_and_names:
            x = stage(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]


class YOLOv3(nn.Module):
    def __init__(self, pretrain_path=None, num_classes=80, num_anchors=3):
        super(YOLOv3, self).__init__()

        self.backbone = Darknet()
        if pretrain_path:
            print("Load backbone checkpoint %s" % pretrain_path)
            state_dict = torch.load(pretrain_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)

        in_channels = [256, 512, 1024]

        out_filter_0 = (1 + 4 + num_classes) * num_anchors
        self.out0 = self._make_embedding(in_channels[-1], [512, 1024], out_filter_0)

        self.out1_cbl = conv_bn_lrelu(512, 256, 1)
        self.out1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        out_filter_1 = (1 + 4 + num_classes) * num_anchors
        self.out1 = self._make_embedding(
            in_channels[-2] + 256, [256, 512], out_filter_1
        )

        self.out2_cbl = conv_bn_lrelu(256, 128, 1)
        self.out2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        out_filter_2 = (1 + 4 + num_classes) * num_anchors
        self.out2 = self._make_embedding(
            in_channels[-3] + 128, [128, 256], out_filter_2
        )

    def _make_embedding(self, in_filters, filters_list, out_filter):
        m = nn.ModuleList(
            [
                conv_bn_lrelu(in_filters, filters_list[0], 1),
                conv_bn_lrelu(filters_list[0], filters_list[1], 3),
                conv_bn_lrelu(filters_list[1], filters_list[0], 1),
                conv_bn_lrelu(filters_list[0], filters_list[1], 3),
                conv_bn_lrelu(filters_list[1], filters_list[0], 1),
                conv_bn_lrelu(filters_list[0], filters_list[1], 3),
            ]
        )
        m.add_module(
            "conv_out",
            nn.Conv2d(
                filters_list[1],
                out_filter,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.out0, x0)
        #  yolo branch 1
        x1_in = self.out1_cbl(out0_branch)
        x1_in = self.out1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.out1, x1_in)
        #  yolo branch 2
        x2_in = self.out2_cbl(out1_branch)
        x2_in = self.out2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.out2, x2_in)

        outputs = [out0, out1, out2]

        return outputs


def yolov3(pretrain_path=None, model_path=None):
    model = YOLOv3(pretrain_path)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {}
        for key, val in state_dict.items():
            if "module." in key:
                key = key.replace("module.", "")
            new_state_dict[key] = val

        model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yolov3")
    parser.add_argument("--pretrain-path", default=None, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    args = parser.parse_args()
    model = yolov3(args.pretrain_path, args.model_path)
    print(model)
