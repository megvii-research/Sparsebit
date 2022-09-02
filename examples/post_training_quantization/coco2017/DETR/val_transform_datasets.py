from pathlib import Path
from detr.datasets.coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms("val"), return_masks=args.masks)
    return dataset

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')