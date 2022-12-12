import argparse
import os
import random
import numpy as np
import shutil
import time
import warnings
from enum import Enum
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sparsebit.quantization import QuantModel, parse_qconfig
from model import efficientnet_lite0
import utils


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("qconfig", help="the path of quant config")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default="imagenet",
    help="path to dataset (default: imagenet)",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="efficientnet_lite0",
    help="model architecture: " + " (default: efficientnet_lite0)",
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
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("-b", "--batch-size", default=256, type=int)
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
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--pretrain_path",
    default="./checkpoints/efficientnet_lite0.pth",
    help="pretrained checkpoint",
)
parser.add_argument("--resume", default="", help="resume from checkpoint")
parser.add_argument(
    "--output_dir", required=True, help="path where to save, empty for no saving"
)
parser.add_argument(
    "--device", default="cuda", help="device to use for training / testing"
)
parser.add_argument("--seed", default=0, type=int)
# distributed training parameters
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)


def main():
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    qconfig = parse_qconfig(args.qconfig)

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    calib_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == "efficientnet_lite0":
            model = efficientnet_lite0(ckpt_path=args.pretrain_path)
        else:
            raise "unsupport arch {}".format(args.arch)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == "efficientnet_lite0":
            model = efficientnet_lite0()
        else:
            raise "unsupport arch {}".format(args.arch)

    model.to(device)
    model = QuantModel(model, config=qconfig)

    # set head and tail of model is 8bit
    model.model.conv_stem.input_quantizer.set_bit(bit=8)
    model.model.conv_stem.weight_quantizer.set_bit(bit=8)
    model.model.classifier.input_quantizer.set_bit(bit=8)
    model.model.classifier.weight_quantizer.set_bit(bit=8)
    # run calibration
    model.prepare_calibration()
    calib_size, cur_size = 256, 0
    model.eval()
    with torch.no_grad():
        for data, target in calib_loader:
            model(data.cuda())
            cur_size += data.shape[0]
            if cur_size >= calib_size:
                break
        model.init_QAT()
    print(model.model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 256.0
    args.lr = linear_scaled_lr

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            best_acc1 = checkpoint["best_acc1"]
        lr_scheduler.step(args.start_epoch)
    else:
        best_acc1 = 0

    if args.eval:
        acc1 = validate(val_loader, model, criterion, args)
        print(
            f"Accuracy of the network on the {len(val_dataset)} test images: {acc1:.1f}%"
        )
        return

    log_stats = {"n_parameters": n_parameters}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        log_stats.update(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch,}
        )
        train(train_loader, model, criterion, optimizer, epoch, args, device)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "best_acc1": best_acc1,
                    },
                    checkpoint_path,
                )

        acc1 = validate(val_loader, model, criterion, args)
        print(
            f"Accuracy of the network on the {len(val_dataset)} test images: {acc1:.1f}%"
        )

        if best_acc1 < acc1:
            best_acc1 = acc1
            if args.output_dir:
                checkpoint_paths = [output_dir / "best_checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "best_acc1": best_acc1,
                        },
                        checkpoint_path,
                    )

        print(f"Best accuracy: {best_acc1:.2f}%")
        log_stats.update({"test_acc1": acc1})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    print("Training is Done, best: {}".format(best_acc1))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(
                json.dumps({"best_acc1": best_acc1, "training_args": str(args)}) + "\n"
            )


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


@torch.no_grad()
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
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
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    print("Total Time: {}".format(time.time() - start))
    return top1.avg.item()


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
