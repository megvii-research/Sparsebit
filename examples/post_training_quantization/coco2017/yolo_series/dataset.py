import os
import copy
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace
from pycocotools.coco import COCO

import torch

from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms


class Augmentor:
    def __init__(
        self,
        image_size=(608, 608),
        min_ious=[0.1, 0.3, 0.5, 0.7, 0.9],
        min_crop_size=0.3,
        expand_ratio=[1, 4],
        expand_means=[123.675, 116.28, 103.53],
        is_train=False,
    ):
        self.image_size = image_size
        self.new_w, self.new_h = self.image_size
        self.min_ious = min_ious
        self.min_crop_size = min_crop_size
        self.expand_ratio = expand_ratio
        self.expand_means = expand_means  # rgb, imagenet_means
        self.is_train = is_train

    def __call__(self, img, annotations, is_expand=True):
        if self.is_train:
            if np.random.rand() < 0.5:  # brightness
                w = np.random.uniform(1 - 32.0 / 255, 1 + 32.0 / 255)
                img = self._blending(img, src_img=0, src_weight=(1 - w), dst_weight=w)
            if np.random.rand() < 0.5:  # contrast
                w = np.random.uniform(0.5, 1.5)
                img = self._blending(
                    img, src_img=img.mean(), src_weight=(1 - w), dst_weight=w
                )
            if np.random.rand() < 0.5:  # saturation
                w = np.random.uniform(0.5, 1.5)
                grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
                img = self._blending(
                    img, src_img=grayscale, src_weight=(1 - w), dst_weight=w
                )
            img, annotations, crop_tlbr = self._miniou_randomcrop(img, annotations)
            if is_expand and np.random.rand() < 0.6:  # expand
                img, annotations = self._expand(img, annotations)
            img, annotations, resize_ratio = self._resize(img, annotations)
            if np.random.rand() < 0.5:
                img, annotations = self._randflip(img, annotations)
        else:
            img, annotations, _ = self._resize(img, annotations)
            crop_tlbr = None
            resize_ratio = None
        return img, annotations, crop_tlbr, resize_ratio

    def _resize(self, img, annotations):
        """
        boxes is a tensor with shape=[:, 4], (tx, ty, w, h)
        """
        h, w, _ = img.shape
        pil_img = Image.fromarray(img)
        if self.is_train:
            interp_method = np.random.choice(
                [Image.BILINEAR, Image.BICUBIC, Image.NEAREST, Image.LANCZOS]
            )
        else:
            interp_method = Image.BILINEAR
        pil_img = pil_img.resize((self.new_w, self.new_h), resample=interp_method)
        resized_img = np.asarray(pil_img)
        resize_ratio = np.array([1.0, 1.0])
        if annotations is not None:
            annotations[:, 0:4:2] = annotations[:, 0:4:2] * (self.new_w * 1.0 / w)
            annotations[:, 1:4:2] = annotations[:, 1:4:2] * (self.new_h * 1.0 / h)
            resize_ratio = np.array([self.new_w * 1.0 / w, self.new_h * 1.0 / h])
        return resized_img, annotations, resize_ratio

    def _randflip(self, img, annotations):
        h, w, c = img.shape
        img = img[:, ::-1, :]
        if annotations is not None:
            annotations[:, 0:4:2] = w - annotations[:, 2::-2]
        return img, annotations

    def _expand(self, img, annotations):
        h, w, c = img.shape
        ratio = np.random.uniform(self.expand_ratio[0], self.expand_ratio[1])
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        expand_img = np.full(
            (int(h * ratio), int(w * ratio), c), self.expand_means
        ).astype(img.dtype)
        expand_img[top : top + h, left : left + w] = img
        if annotations is not None:
            annotations[:, 0:4:2] += left
            annotations[:, 1:4:2] += top
        return expand_img, annotations

    def _blending(self, img, src_img, src_weight, dst_weight):
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = src_weight * src_img + dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return src_weight * src_img + dst_weight * img

    def _miniou_randomcrop(self, img, annotations):
        if annotations is None:
            return img, annotations, np.array([0, 0, 0, 0])
        sample_mode = (1, *self.min_ious, 0)
        h, w = img.shape[:2]
        boxes = annotations[:, :4].copy()
        while True:
            min_iou = np.random.choice(sample_mode)
            if min_iou == 1:
                return img, annotations, np.array([0, 0, 0, 0])
            for i in range(50):
                new_w = int(np.random.uniform(self.min_crop_size * w, w))
                new_h = int(np.random.uniform(self.min_crop_size * h, h))
                if new_h / new_w < 0.5 or new_h / new_w > 2:  # h/w in [0.5, 2]
                    continue
                left = int(np.random.uniform(w - new_w))
                top = int(np.random.uniform(h - new_h))
                patch = np.array([left, top, left + new_w, top + new_h])
                overlaps = pairwise_iou(patch.reshape(-1, 4), boxes.reshape(-1, 4))
                if overlaps.min() < min_iou:
                    continue
                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (
                    (center[:, 0] > patch[0])
                    * (center[:, 1] > patch[1])
                    * (center[:, 0] < patch[2])
                    * (center[:, 1] < patch[3])
                )
                if not mask.any():
                    continue
                # crop image
                img = img[
                    top : top + new_h, left : left + new_w, :
                ]  # [:, :, ::-1] # h, w, c
                tl = np.array([left, top])
                br = np.array([left + new_w, top + new_h])
                boxes[:, :2] = np.maximum(boxes[:, :2], tl) - tl
                boxes[:, 2:] = np.minimum(boxes[:, 2:], br) - tl
                annotations[:, :4] = boxes
                crop_tlbr = np.array([left, top, w - new_w - left, h - new_h - top])
                return img, annotations, crop_tlbr


class COCODataset(Dataset):
    def __init__(
        self,
        dataset_root="./datasets",
        dataset_name="coco_2017_val",
        data_format="rgb",
        image_size=(608, 608),
        input_norm=True,
        mosaic=False,
        mosaic_min_offset=0.2,
        mosaic_image_num=4,
        is_train=False,
    ):
        super(COCODataset, self).__init__()

        dataset_split = {
            "coco_2017_train": (
                "train2017",
                "annotations/instances_train2017.json",
            ),
            "coco_2017_val": (
                "val2017",
                "annotations/instances_val2017.json",
            ),
        }

        image_root, json_file = dataset_split[dataset_name]
        self.image_root = os.path.join(dataset_root, image_root)
        self.json_file = os.path.join(dataset_root, json_file)

        self.meta = {}
        self.dataset_dicts = self._load_annotations()

        self.is_train = is_train

        if self.is_train:
            self.dataset_dicts = self._filter_annotations(self.dataset_dicts)
            self._set_group_flag()

        self.data_format = data_format

        self.augmentor = Augmentor(image_size, is_train=self.is_train)
        self.to_tensor = tv_transforms.ToTensor()

        if input_norm:
            self.normalize = tv_transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        else:
            self.normalize = tv_transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        self.mosaic = mosaic
        self.mosaic_min_offset = mosaic_min_offset
        self.mosaic_image_num = mosaic_image_num

    def _filter_annotations(self, dataset_dicts):
        num_before = len(dataset_dicts)

        def valid(anns):
            for ann in anns:
                if ann.get("iscrowd", 0) == 0:
                    return True
            return False

        dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
        num_after = len(dataset_dicts)
        print(
            "Removed {} images with no usable annotations. {} images left.".format(
                num_before - num_after, num_after
            )
        )
        return dataset_dicts

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        if "width" in self.dataset_dicts[0] and "height" in self.dataset_dicts[0]:
            for i in range(len(self)):
                dataset_dict = self.dataset_dicts[i]
                if dataset_dict["width"] / dataset_dict["height"] > 1:
                    self.aspect_ratios[i] = 1

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_and_aug(self, dataset_dict, is_expand=True):
        image = np.asarray(
            (Image.open(dataset_dict["file_name"])).convert("RGB")
        )  # rgb
        if (self.data_format).lower() == "bgr":
            image = image[:, :, ::-1]
        # check image_size
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        assert (
            image_wh == expected_wh
        ), "Mismatched (W,H), name: {}, got {}, expect {}".format(
            dataset_dict["file_name"] if "file_name" in dataset_dict else "",
            image_wh,
            expected_wh,
        )
        if "width" not in dataset_dict:
            dataset_dict["width"] = image.shape[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.shape[0]
        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        if len(annotations) == 0:
            annotations = None

        # apply transfrom
        if annotations is not None:
            boxes = np.array([obj["bbox"] for obj in annotations])
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            classes = np.array([obj["category_id"] for obj in annotations])[
                :, np.newaxis
            ]
            annotations = np.concatenate([boxes, classes], axis=1)

        image, annotations, crop_tlbr, resize_ratio = self.augmentor(
            image, annotations, is_expand=is_expand
        )

        return image, annotations, crop_tlbr, resize_ratio

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances."""
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        if self.is_train and self.mosaic and np.random.rand() < 0.5:
            cfg_w, cfg_h = self.augmentor.image_size
            out_image = np.zeros([cfg_h, cfg_w, 3])
            min_offset = self.mosaic_min_offset
            out_annotations = []
            cut_x = random.randint(
                int(cfg_w * min_offset), int(cfg_w * (1 - min_offset))
            )
            cut_y = random.randint(
                int(cfg_h * min_offset), int(cfg_h * (1 - min_offset))
            )

            for i in range(self.mosaic_image_num):
                if i != 0:
                    dataset_dict = copy.deepcopy(random.choice(self.dataset_dicts))
                image, annotations, crop_tlbr, resize_ratio = self._load_and_aug(
                    dataset_dict, is_expand=False
                )

                left_shift = int(min(cut_x, crop_tlbr[0] * resize_ratio[0]))
                top_shift = int(min(cut_y, crop_tlbr[1] * resize_ratio[1]))
                right_shift = int(min(cfg_w - cut_x, crop_tlbr[2] * resize_ratio[0]))
                bot_shift = int(min(cfg_h - cut_y, crop_tlbr[3] * resize_ratio[1]))

                out_image, out_annotation = blend_truth_mosaic(
                    out_image,
                    image,
                    annotations.copy(),
                    cfg_w,
                    cfg_h,
                    cut_x,
                    cut_y,
                    i,
                    left_shift,
                    right_shift,
                    top_shift,
                    bot_shift,
                )
                out_annotations.append(out_annotation)

            out_annotations = np.concatenate(out_annotations, axis=0)

            image = out_image.astype(np.uint8)
            annotations = out_annotations

        else:
            image, annotations, _, _ = self._load_and_aug(dataset_dict)

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w
            # filter empty target
            widths = annotations[:, 2] - annotations[:, 0]
            heights = annotations[:, 3] - annotations[:, 1]
            annotations = annotations[(widths > 0) & (heights > 0)]
            annotations = torch.from_numpy(annotations)
        dataset_dict["image"] = self.normalize(self.to_tensor(Image.fromarray(image)))
        dataset_dict["annotations"] = annotations
        return dataset_dict

    def _load_annotations(self):
        """
        Load a json file with COCO's instances annotation format.
        """

        image_root = self.image_root
        json_file = self.json_file

        coco_api = COCO(json_file)

        meta = self.meta
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta["thing_classes"] = thing_classes
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta["thing_dataset_id_to_contiguous_id"] = id_map
        meta["json_file"] = json_file
        self.meta = meta

        img_ids = sorted(coco_api.imgs.keys())

        imgs = coco_api.loadImgs(img_ids)

        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

        imgs_anns = list(zip(imgs, anns))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "category_id"]

        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(
        np.where(
            ((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy))
            | ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx))
            | ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0))
            | ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0))
        )[0]
    )
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(
    out_img,
    img,
    bboxes,
    w,
    h,
    cut_x,
    cut_y,
    i_mixup,
    left_shift,
    right_shift,
    top_shift,
    bot_shift,
):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[
            top_shift : top_shift + cut_y, left_shift : left_shift + cut_x
        ]
    if i_mixup == 1:
        bboxes = filter_truth(
            bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0
        )
        out_img[:cut_y, cut_x:] = img[
            top_shift : top_shift + cut_y, cut_x - right_shift : w - right_shift
        ]
    if i_mixup == 2:
        bboxes = filter_truth(
            bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y
        )
        out_img[cut_y:, :cut_x] = img[
            cut_y - bot_shift : h - bot_shift, left_shift : left_shift + cut_x
        ]
    if i_mixup == 3:
        bboxes = filter_truth(
            bboxes,
            cut_x - right_shift,
            cut_y - bot_shift,
            w - cut_x,
            h - cut_y,
            cut_x,
            cut_y,
        )
        out_img[cut_y:, cut_x:] = img[
            cut_y - bot_shift : h - bot_shift, cut_x - right_shift : w - right_shift
        ]

    return out_img, bboxes


class ConcatDataset(Dataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        if hasattr(self.datasets[0], "aspect_ratios"):
            aspect_ratios = [d.aspect_ratios for d in self.datasets]
            self.aspect_ratios = np.concatenate(aspect_ratios)
        if hasattr(self.datasets[0], "meta"):
            self.meta = {}
            for d in self.datasets:
                self.meta.update(d.meta)
            self.meta = SimpleNamespace(**self.meta)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def collate_fn(batch):
    images_id = []
    images = []
    annotations_list = []
    images_size = []
    for _, sample in enumerate(batch):
        images_id.append(sample["image_id"])
        images.append(sample["image"])
        images_size.append((sample["width"], sample["height"]))
        annotations_list.append(sample["annotations"])
    return {
        "data": torch.stack(images, 0),
        "targets": annotations_list,
        "images_id": images_id,
        "images_size": images_size,
    }


def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    width_height = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.maximum(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]
    width_height = np.maximum(width_height, 0)
    inter = width_height.prod(axis=2)  # [N,M]
    del width_height
    # handle empty boxes
    iou = np.where(inter > 0, inter / (area1[:, None] + area2 - inter), 0)
    return iou


def coco_dataset(
    dataset_root="./datasets",
    dataset_names=["coco_2017_val"],
    data_format="rgb",
    image_size=(608, 608),
    input_norm=True,
    mosaic=False,
    mosaic_min_offset=0.2,
    mosaic_image_num=4,
    is_train=False,
):
    sub_datasets = [
        COCODataset(
            dataset_root=dataset_root,
            dataset_name=name,
            data_format=data_format,
            image_size=image_size,
            input_norm=input_norm,
            mosaic=mosaic,
            mosaic_min_offset=mosaic_min_offset,
            mosaic_image_num=mosaic_image_num,
            is_train=is_train,
        )
        for name in dataset_names
    ]
    dataset = ConcatDataset(sub_datasets)
    return dataset


if __name__ == "__main__":
    import cv2

    dataset = coco_dataset()
    for i in range(100):
        item = dataset.__getitem__(i)
        img = item["image"].numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0  # rgb -> bgr
        annotations = item["annotations"].numpy()
        print(annotations)
        for ann in annotations:
            tx, ty, bx, by = list(map(int, ann[:4]))
            cls_label = dataset.meta.thing_classes[int(ann[4])]
            img = img.copy().astype(np.uint8)
            img = cv2.rectangle(img, (tx, ty), (bx, by), color=(0, 0, 255))
            cv2.putText(
                img, cls_label, (tx, ty), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0)
            )
            cv2.imwrite("data.png", img)
            import ipdb

            ipdb.set_trace()
