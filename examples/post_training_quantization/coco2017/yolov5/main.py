# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0])+"/yolov5"  # YOLOv5 root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH

from models import yolov5n, yolov5s
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from yolov5.utils.plots import output_to_target, plot_images, plot_val_study
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from sparsebit.quantization import QuantModel, parse_qconfig


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@torch.no_grad()
def main(args):
    if args.model_name == "yolov5n":
        model = yolov5n(model_path="checkpoints/yolov5n.pth")
    elif args.model_name == "yolov5s":
        model = yolov5s(model_path="checkpoints/yolov5s.pth")
    else:
        raise NotImplementedError

    qconfig = parse_qconfig(args.qconfig_path)
    qmodel = QuantModel(model.model4quant, config=qconfig)

    setattr(model,"model4quant", qmodel)
    imgsz = 640
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure
    model.eval()
    model.cuda()
    
    cuda = device.type != 'cpu'
    nc = 80  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    task = "val" # path to val images
    dataloader = create_dataloader(os.path.join(args.data_path,"val2017.txt"),
                                    imgsz,
                                    1,
                                    32,
                                    False,
                                    pad=0.5,
                                    rect=True,
                                    workers=args.workers,
                                    prefix=colorstr(f'{task}: '))[0]

    # Calibration
    calib_loader = create_dataloader(os.path.join(args.data_path,"calib2017.txt"),
                                       imgsz,
                                       1,
                                       32,
                                       False,
                                       pad=0.5,
                                       rect=True,
                                       workers=args.workers,
                                       prefix=colorstr(f'{task}: '))[0]

    qmodel.prepare_calibration()
    for batch_meta in calib_loader:
        data = batch_meta[0]/255
        with torch.no_grad():
            _ = qmodel(data.to(device, non_blocking=True))
    qmodel.calc_qparams()
    qmodel.set_quant(w_quant=True, a_quant=True)

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    p, r, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    stats, ap = [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for im, targets, paths, shapes in pbar:
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds = model(im)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        with dt[2]:
            preds = non_max_suppression(preds,
                                        args.conf_thres,
                                        args.iou_thres,
                                        labels=[],
                                        multi_label=True,
                                        agnostic=False,
                                        max_det=300)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            npr = pred.shape[0]  # number of labels, predictions
            shape = shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            # Predictions
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="yolov5n", choices=["yolov5n", "yolov5s"])
    parser.add_argument('--qconfig_path', type=str, default="./qconfig.yaml")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU threshold')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
