import os
import cv2
import numpy as np
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nori2 as nori


def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)

    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image

    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)

    assert img is not None, "failed to decode"
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img


class DATASET(Dataset):
    def __init__(self, transform, mode, size: int = None, dataset_name=None):
        assert dataset_name is None, "imagenet is incompatible with SUBDATASETS"
        assert mode in [
            "train",
            "validation",
        ], "only support mode equals train or validation"
        self.instance_per_epoch = {
            "train": 5000 * 256,
            "validation": 50000,
        }[mode]
        if size is not None:
            self.instance_per_epoch = min(self.instance_per_epoch, size)
        nfname = {
            "train": "imagenet.train.nori.list",
            "validation": "imagenet.val.nori.list",
        }[mode]
        self.nid_filename = "s3://hw-share/dataset/imagenet/ILSVRC2012/" + nfname
        self.nf = nori.Fetcher()
        self.nid_labels = self.load()
        self.train = mode == "train"
        self.transform = transform
        self.nb_classes = 1000
        self.dataset_size = size

    def load(self):
        self.nid_labels = []
        with nori.smart_open(self.nid_filename) as f:
            for line in f:
                nid, label, _ = line.strip().split("\t")
                self.nid_labels.append((nid, int(label)))
        return self.nid_labels[: self.instance_per_epoch]

    def __len__(self):
        origin_len = self.instance_per_epoch
        if isinstance(self.dataset_size, int):
            return min(self.dataset_size, origin_len)
        else:
            return origin_len

    def __getitem__(self, idx):
        nid, label = self.nid_labels[idx]
        data = self.nf.get(nid)
        img = imdecode(data)[:, :, :3][:, :, ::-1]  # bgr -> rgb
        img = Image.fromarray(img)
        img = self.transform(img.copy())
        label = torch.tensor(label, dtype=torch.long)
        return img, label


if __name__ == "__main__":
    import datasets

    dataset = DATASET(transform=None, mode="validation")
    img, label = dataset.__getitem__(0)
