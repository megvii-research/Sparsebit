import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model import decode_outputs, postprocess, postprocess_boxes, yolov3 as build_model
from dataset import coco_dataset as build_dataset, collate_fn
from evaluate import coco_evaluate


from sparsebit.quantization import QuantModel, parse_qconfig


def main(args):

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")

    # Build model
    model = build_model()
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # Convert to quant model
    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig)
    qmodel.eval()
    qmodel = qmodel.to(device)

    # Create dataset
    train_dataset = build_dataset(
        dataset_root=args.dataset_root,
        dataset_names=[args.train_datasets],
        data_format=args.input_format,
        image_size=(args.image_size, args.image_size),
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

    # Calibration
    qmodel.prepare_calibration()
    calibration_size = 8
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
    qmodel.eval()

    # Evaluate
    all_iters = len(val_loader)
    processed_results = []
    for i, batch_meta in enumerate(val_loader):
        batch_data = batch_meta["data"].to(device)
        batch_id = batch_meta["images_id"]
        batch_size = batch_meta["images_size"]
        with torch.no_grad():
            outputs = qmodel(batch_data)

        predictions = decode_outputs(
            outputs, (args.image_size, args.image_size), args.num_classes
        )
        detections = postprocess(
            predictions, args.num_classes, args.conf_threshold, args.nms_threshold
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
        processed_results, val_dataset.meta, iou_type="bbox"
    )
    print(
        "AP:%.4lf, AP50:%.4lf, AP75:%.4lf"
        % (results["AP"], results["AP50"], results["AP75"])
    )

    # Export onnx
    qmodel.export_onnx(data, name="qyolov3.onnx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--conf-threshold", default=0.01, type=float, help="confidence threshold"
    )
    parser.add_argument(
        "--nms-threshold", default=0.5, type=float, help="nms threshold"
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, help="the frequency of logging"
    )
    parser.add_argument("--qconfig", default="./qconfig.yaml", type=str)
    args = parser.parse_args()
    main(args)
