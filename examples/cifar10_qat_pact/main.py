import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

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


if not torch.cuda.is_available():
    raise NotImplementedError("This example should run on a GPU device.")  # 确定在GPU上运行


config = "qconfig.yaml"  # QAT配置文件——包括量化方式（dorefa/lsq），权重和激活值的量化bit数等
workers = 4
epochs = 200
start_epoch = 0
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
pretrained = ""
qconfig = parse_qconfig(config)


model = resnet20(num_classes=10)  # 以resnet20作为基础模型
if pretrained:  # 可以采用pretrained中保存的模型参数
    ckpt_state_dict = torch.load(pretrained)
    model.load_state_dict(ckpt_state_dict)

cudnn.benchmark = True


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, 4),  # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # 指定各通道均值和标准差，将数据归一化
    ]
)

trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
)

testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
)


model = QuantModel(model, qconfig).cuda()  # 将model转化为量化模型，以支持后续QAT的各种量化操作


model.prepare_calibration()  # 进入calibration状态
calib_size, cur_size = 256, 0
# 在eval模式且无需计算梯度的条件下用训练集进行calibrate
model.eval()
with torch.no_grad():
    for data, target in trainloader:
        model(data.cuda())
        cur_size += data.shape[0]
        if cur_size >= calib_size:
            break
model.init_QAT()  # 调用API，初始化QAT
model.set_lastmodule_wbit(bit=8)  # 额外规定最后一层权重的量化bit数
print(model.model)  # 可以在print出的模型信息中看到网络各层weight和activation的量化scale和zeropoint

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr,
    momentum=momentum,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], last_epoch=start_epoch - 1
)


# PACT算法中对 alpha 增加 L2-regularization
def get_l2_loss(model, scale=0.0001):
    l2_loss = 0
    for n, p in model.named_parameters():
        if "alpha" in n:
            l2_loss += (p ** 2).sum()
    return l2_loss * scale


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    cross_losses = AverageMeter("CrossLoss", ":.4e")
    l2_losses = AverageMeter("L2Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, cross_losses, l2_losses, losses, top1],
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
        cross_loss = criterion(output, target)
        l2_loss = get_l2_loss(model)
        loss = cross_loss + l2_loss

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        cross_losses.update(cross_loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        l2_losses.update(l2_loss.item(), images.size(0))
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


def validate(val_loader, model, criterion):
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


best_acc1 = 0
for epoch in range(start_epoch, epochs):
    # train for one epoch
    train(
        trainloader,
        model,
        criterion,
        optimizer,
        epoch,
    )

    # evaluate on validation set
    acc1 = validate(testloader, model, criterion)

    scheduler.step()

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        is_best,
    )

print("Training is Done, best: {}".format(best_acc1))

# export onnx
model.eval()
with torch.no_grad():
    model.export_onnx(torch.randn(1, 3, 32, 32), name="qresnet20.onnx", extra_info=True)
