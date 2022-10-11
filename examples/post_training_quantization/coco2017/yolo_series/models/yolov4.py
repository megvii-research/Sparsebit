import torch
import torch.nn as nn


class Conv_Bn_Activation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation,
        bn=True,
        bias=False,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias)
        )

        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))

        if activation == "mish":
            self.conv.append(nn.Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, "mish"))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, "mish")
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, "mish")
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, "mish")
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, "mish")
        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, "mish")
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, "mish")
        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, "mish")
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], 1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, "mish")
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, "mish")
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, "mish")
        self.resblock = ResBlock(ch=64, nblocks=2)
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, "mish")
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, "mish")
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, "mish")
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, "mish")
        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, "mish")
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, "mish")
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, "mish")
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, "mish")
        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, "mish")
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, "mish")
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, "mish")
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, "mish")
        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, "mish")
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, "leaky")
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], 1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        up = self.upsample1(x7)
        x8 = self.conv8(downsample4)
        x8 = torch.cat([x8, up], 1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        up = self.upsample2(x14)
        x15 = self.conv15(downsample3)
        x15 = torch.cat([x15, up], 1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv2 = Conv_Bn_Activation(
            256, output_ch, 1, 1, "linear", bn=False, bias=True
        )
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, "leaky")
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv10 = Conv_Bn_Activation(
            512, output_ch, 1, 1, "linear", bn=False, bias=True
        )
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, "leaky")
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv18 = Conv_Bn_Activation(
            1024, output_ch, 1, 1, "linear", bn=False, bias=True
        )

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        x3 = self.conv3(input1)
        x3 = torch.cat([x3, input2], 1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x8)
        x11 = torch.cat([x11, input3], 1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        return x18, x10, x2


class Yolov4(nn.Module):
    def __init__(self, n_classes=80):
        super().__init__()
        output_ch = (4 + 1 + n_classes) * 3

        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

        self.neck = Neck()

        self.head = Yolov4Head(output_ch, n_classes)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output


def yolov4(pretrain_path=None, model_path=None):
    model = Yolov4()
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
    import ipdb

    parser = argparse.ArgumentParser(description="Yolov4")
    parser.add_argument("--pretrain-path", default=None, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--image-size", default=416, type=int)
    args = parser.parse_args()
    model = yolov4(args.pretrain_path, args.model_path)
    print(model)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, args.image_size, args.image_size))
        ipdb.set_trace()
