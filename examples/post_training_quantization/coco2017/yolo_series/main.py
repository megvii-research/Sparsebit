import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models
from utils import decode_outputs
from dataset import coco_dataset as build_dataset, collate_fn
from evaluate import coco_evaluate

from sparsebit.quantization import QuantModel, parse_qconfig


def main(args):

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")

    # Build model
    model = models.__dict__[args.arch](model_path=args.model_path)

    # Create dataset
    train_dataset = build_dataset(
        dataset_root=args.dataset_root,
        dataset_names=[args.train_datasets],
        data_format=args.input_format,
        image_size=(args.image_size, args.image_size),
        input_norm=not args.wo_input_norm,
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataset = build_dataset(
        dataset_root=args.dataset_root,
        dataset_names=[args.val_datasets],
        data_format=args.input_format,
        image_size=(args.image_size, args.image_size),
        input_norm=not args.wo_input_norm,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Convert to quant model
    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig)
    qmodel.eval()
    qmodel = qmodel.to(device)

    # Calibration
    qmodel.prepare_calibration()
    calibration_size = args.calib_size
    cur_size = 0
    for i, batch_meta in enumerate(train_loader):
        data = batch_meta["data"]
        with torch.no_grad():
            _ = qmodel(data.to(device))
        cur_size += data.shape[0]
        if cur_size >= calibration_size:
            break
    qmodel.calc_qparams()

    qmodel.set_quant(w_quant=True, a_quant=True)

    # Evaluate
    validate(qmodel, val_loader, val_dataset.meta, device, args)

    # Export onnx
    qmodel.export_onnx(data, name="q%s.onnx" % (args.arch))


def validate(model, val_loader, meta, device, args):

    model.eval()

    all_iters = len(val_loader)
    processed_results = []
    for i, batch_meta in enumerate(val_loader):
        batch_data = batch_meta["data"].to(device)
        batch_id = batch_meta["images_id"]
        batch_size = batch_meta["images_size"]

        with torch.no_grad():
            outputs = model(batch_data)

        detections = decode_outputs(
            outputs,
            args.arch,
            (args.image_size, args.image_size),
            args.num_classes,
            args.conf_threshold,
            args.nms_threshold,
        )

        for ii, det in enumerate(detections):
            image_id = batch_id[ii]
            if det is None:
                continue
            det = det.cpu().detach().numpy()
            boxes = postprocess_boxes(
                det[:, :4],
                src_size=batch_size[ii],
                eval_size=(args.image_size, args.image_size),
            )

            pred_result = {}
            pred_result["image_id"] = image_id
            pred_result["instances"] = np.concatenate(
                [boxes, det[:, 4:5] * det[:, 5:6], det[:, 6:7]], axis=-1
            )
            processed_results.append(pred_result)

        if (i + 1) % args.print_freq == 0:
            print("{}/{}".format((i + 1), all_iters))

    results, results_per_category = coco_evaluate(
        processed_results, meta, iou_type="bbox"
    )
    print(
        "AP:%.4lf, AP50:%.4lf, AP75:%.4lf"
        % (results["AP"], results["AP50"], results["AP75"])
    )


def postprocess_boxes(pred_bbox, src_size, eval_size):
    pred_coor = pred_bbox
    src_w, src_h = src_size
    eval_w, eval_h = eval_size
    resize_ratio_w = float(eval_w) / src_w
    resize_ratio_h = float(eval_h) / src_h
    dw = (eval_size[0] - resize_ratio_w * src_w) / 2
    dh = (eval_size[1] - resize_ratio_h * src_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio_w
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio_h
    pred_coor = np.concatenate(
        [
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [src_w - 1, src_h - 1]),
        ],
        axis=-1,
    )
    return pred_coor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="yolov3",
        choices=["yolov3", "yolov4"],
        help="model architecture",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Use cpu inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Trained state_dict file path to open",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./coco",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        default="coco_2017_train",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        default="coco_2017_val",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="rgb",
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of workers used in dataloading"
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--image-size", default=608, type=int)
    parser.add_argument("--num-classes", default=80, type=int)
    parser.add_argument("--wo-input-norm", action="store_true")
    parser.add_argument(
        "--conf-threshold", default=0.01, type=float, help="confidence threshold"
    )
    parser.add_argument(
        "--nms-threshold", default=0.5, type=float, help="nms threshold"
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, help="the frequency of logging"
    )
    parser.add_argument(
        "--calib-size", default=16, type=int, help="the frequency of logging"
    )
    parser.add_argument("--qconfig", default="./qconfig.yaml", type=str)
    args = parser.parse_args()
    main(args)
