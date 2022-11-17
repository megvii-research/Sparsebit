import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import resnet20
from sparsebit.quantization import QuantModel, parse_qconfig
from sparsebit.quantization.regularizers import build_regularizer


parser = argparse.ArgumentParser(description="PyTorch Cifar Training")
parser.add_argument("config", help="the path of quant config")
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--pretrained", default=None, type=str, help="use pre-trained model"
)
parser.add_argument(
    "--regularizer-lambda", default=1e-4, type=float, help="regularizer lambda"
)
parser.add_argument("--calib-size", default=256, type=int, help="calibration size")


def main():
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "This example should run on a GPU device."
        )  # 确定在GPU上运行

    model = resnet20(num_classes=10)  # 以resnet20作为基础模型
    if args.pretrained:  # 可以采用pretrained中保存的模型参数
        ckpt_state_dict = torch.load(args.pretrained)
        model.load_state_dict(ckpt_state_dict)

    cudnn.benchmark = True

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(32, 4),  # 随机裁剪
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # 指定各通道均值和标准差，将数据归一化
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # 指定各通道均值和标准差，将数据归一化
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    qconfig = parse_qconfig(args.config)

    qmodel = QuantModel(model, qconfig).cuda()  # 将model转化为量化模型，以支持后续QAT的各种量化操作

    # set head and tail of model is 8bit
    qmodel.model.conv1.input_quantizer.set_bit(bit=8)
    qmodel.model.conv1.weight_quantizer.set_bit(bit=8)
    qmodel.model.fc.input_quantizer.set_bit(bit=8)
    qmodel.model.fc.weight_quantizer.set_bit(bit=8)

    qmodel.prepare_calibration()  # 进入calibration状态
    calib_size, cur_size = args.calib_size, 0
    # 在eval模式且无需计算梯度的条件下用训练集进行calibrate
    qmodel.eval()
    with torch.no_grad():
        for data, target in trainloader:
            qmodel(data.cuda())
            cur_size += data.shape[0]
            if cur_size >= calib_size:
                break
        qmodel.init_QAT()  # 调用API，初始化QAT
    print(qmodel.model)  # 可以在print出的模型信息中看到网络各层weight和activation的量化scale和zeropoint

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        qmodel.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1
    )

    if qconfig.REGULARIZER.ENABLE:
        regularizer = build_regularizer(qconfig)
    else:
        regularizer = None

    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(
            trainloader,
            qmodel,
            criterion,
            optimizer,
            epoch,
            regularizer,
            args.regularizer_lambda,
            args.print_freq,
        )

        # evaluate on validation set
        acc1 = validate(testloader, qmodel, criterion, args.print_freq)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": qmodel.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
        )

    print("Training is Done, best: {}".format(best_acc1))

    # export onnx
    qmodel.eval()
    with torch.no_grad():
        qmodel.export_onnx(
            torch.randn(1, 3, 32, 32), name="qresnet20.onnx", extra_info=True
        )


# PACT算法中对 alpha 增加 L2-regularization
def get_pact_regularizer_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        if "alpha" in n:
            loss += (p**2).sum()
    return loss


def get_regularizer_loss(model, is_pact, scale=0):
    if is_pact:
        return get_pact_regularizer_loss(model) * scale
    else:
        return torch.tensor(0.0).cuda()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    regularizer,
    regularizer_lambda,
    print_freq,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    ce_losses = AverageMeter("CELoss", ":.4e")
    regular_losses = AverageMeter("RegularLoss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses, regular_losses, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        ce_loss = criterion(output, target)
        regular_loss = get_regularizer_loss(model, regularizer, regularizer_lambda)
        loss = ce_loss + regular_loss

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        ce_losses.update(ce_loss.item(), images.size(0))
        regular_losses.update(regular_loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        start = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        progress.display_summary()

    print("Total Time: {}".format(time.time() - start))
    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
