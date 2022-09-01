import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sparsebit.quantization import QuantModel, parse_qconfig

from evaluation import validate


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("qconfig", help="the path of quant config")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="deit_tiny_patch16_224",
    help="ViT model architecture. (default: deit_tiny)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)


def main():
    args = parser.parse_args()

    # get pretrained model from https://github.com/facebookresearch/deit
    model = torch.hub.load(
        "facebookresearch/deit:main",
        args.arch,
        pretrained=True,
    )

    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig)

    if torch.cuda.is_available():
        model.cuda()
        qmodel.cuda()

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    calib_dataset = datasets.ImageFolder("calibration_data", transform=train_transforms)
    val_dataset = datasets.ImageFolder("validation_data", transform=val_transforms)
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    qmodel.prepare_calibration()
    # forward calibration-set
    calibration_size = 256
    cur_size = 0
    for data, _ in calib_loader:
        qmodel(data.cuda())
        cur_size += data.shape[0]
        if cur_size >= calibration_size:
            break
    qmodel.calc_qparams()

    criterion = nn.CrossEntropyLoss().cuda()
    qmodel.set_quant(w_quant=True, a_quant=True)

    # validate(val_loader, model, criterion, args)
    validate(val_loader, qmodel, criterion, args)

    qmodel.export_onnx(torch.randn(args.batch_size, 3, 224, 224), name="qViT.onnx")


if __name__ == "__main__":
    main()
