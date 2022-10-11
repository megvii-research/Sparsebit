import os
import json
import itertools
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(outputs, meta, iou_type="bbox", output_dir=None):
    coco_api = COCO(meta.json_file)

    predictions = []
    for output in outputs:
        prediction = instances_to_coco_json(output["instances"], output["image_id"])
        predictions.append(prediction)
    coco_results = list(itertools.chain(*[x for x in predictions]))

    # unmap the category ids for COCO
    reverse_id_mapping = {
        v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()
    }

    for result in coco_results:
        category_id = result["category_id"]
        assert (
            category_id in reverse_id_mapping
        ), "A prediction has category_id={}, which is not available in the dataset.".format(
            category_id
        )
        result["category_id"] = reverse_id_mapping[category_id]

    if output_dir:
        file_path = os.path.join(output_dir, "coco_instances_results.json")
        with open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

    coco_eval = (
        _evaluate_predictions_on_coco(coco_api, coco_results, iou_type=iou_type)
        if len(coco_results) > 0
        else None  # cocoapi does not handle empty results very well
    )
    results, results_per_category = _derive_coco_results(
        coco_eval, iou_type=iou_type, class_names=meta.thing_classes
    )
    return results, results_per_category


def _derive_coco_results(coco_eval, iou_type="bbox", class_names=None):
    """
    Derive the desired score numbers from summarized COCOeval.

    Args:
        coco_eval (None or COCOEval): None represents no predictions from model.
        iou_type (str): specific evaluation task,
            optional values are: "bbox", "segm", "keypoints".
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """

    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }[iou_type]

    if coco_eval is None:
        print("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    # the standard metrics
    results = {
        metric: float(
            coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
        )
        for idx, metric in enumerate(metrics)
    }

    if class_names is None:
        return results, None
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    results_per_category = {}
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category[name] = float(ap * 100)

    # tabulate it

    return results, results_per_category


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances[:, 0:4]
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    boxes = boxes.tolist()
    scores = instances[:, 4].tolist()
    classes = instances[:, 5].tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type="bbox",
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()

    coco_eval.summarize()
    return coco_eval
