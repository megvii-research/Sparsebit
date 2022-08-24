from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat


def conv_bn_relu(ni: int, nf: int, ks: int = 3, stride: int = 1) -> nn.Sequential:
    "Create a seuence Conv2d->BatchNorm2d->ReLu layer."
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        ni,
                        nf,
                        kernel_size=ks,
                        bias=False,
                        stride=stride,
                        padding=ks // 2,
                    ),
                ),
                ("bn", nn.BatchNorm2d(nf)),
                ("relu", nn.ReLU(inplace=True)),
            ]
        )
    )


class ResLayer(nn.Module):
    "Resnet style layer with `ni` inputs."

    def __init__(self, ni: int):
        super(ResLayer, self).__init__()
        self.layer1 = conv_bn_relu(ni, ni // 2, ks=1)
        self.layer2 = conv_bn_relu(ni // 2, ni, ks=3)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Darknet(nn.Module):
    def make_group_layer(self, ch_in: int, num_blocks: int, stride: int = 1):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        return [conv_bn_relu(ch_in, ch_in * 2, stride=stride)] + [
            (ResLayer(ch_in * 2)) for i in range(num_blocks)
        ]

    def __init__(self, depth=53, ch_in=3, nf=32):
        """
        depth (int): depth of darknet used in model, usually use [21, 53] for this param
        ch_in (int): input channels, for example, ch_in of RGB image is 3
        nf (int): number of filters output in stem.
        out_features (List[str]): desired output layer name.
        num_classes (int): For ImageNet, num_classes is 1000. If None, no linear layer will be
            added.
        """
        super(Darknet, self).__init__()
        self.stem = conv_bn_relu(ch_in, nf, ks=3, stride=1)

        current_stride = 1

        "create darknet with `nf` and `num_blocks` layers"
        self.stages_and_names = []
        num_blocks = [1, 2, 8, 8, 4]

        for i, nb in enumerate(num_blocks):
            stage = nn.Sequential(*self.make_group_layer(nf, nb, stride=2))
            name = "dark" + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            current_stride *= 2
            nf *= 2

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        for stage, _ in self.stages_and_names:
            x = stage(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]


class YOLOv3(nn.Module):
    def __init__(self, pretrain_path=None, num_classes=80, num_anchors=3):
        super(YOLOv3, self).__init__()

        self.backbone = Darknet()
        if pretrain_path:
            print("Load backbone checkpoint %s" % pretrain_path)
            state_dict = torch.load(pretrain_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)

        in_channels = [256, 512, 1024]

        out_filter_0 = (1 + 4 + num_classes) * num_anchors
        self.out0 = self._make_embedding(in_channels[-1], [512, 1024], out_filter_0)

        self.out1_cbl = conv_bn_relu(512, 256, 1)
        self.out1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        out_filter_1 = (1 + 4 + num_classes) * num_anchors
        self.out1 = self._make_embedding(
            in_channels[-2] + 256, [256, 512], out_filter_1
        )

        self.out2_cbl = conv_bn_relu(256, 128, 1)
        self.out2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        out_filter_2 = (1 + 4 + num_classes) * num_anchors
        self.out2 = self._make_embedding(
            in_channels[-3] + 128, [128, 256], out_filter_2
        )

    def _make_embedding(self, in_filters, filters_list, out_filter):
        m = nn.ModuleList(
            [
                conv_bn_relu(in_filters, filters_list[0], 1),
                conv_bn_relu(filters_list[0], filters_list[1], 3),
                conv_bn_relu(filters_list[1], filters_list[0], 1),
                conv_bn_relu(filters_list[0], filters_list[1], 3),
                conv_bn_relu(filters_list[1], filters_list[0], 1),
                conv_bn_relu(filters_list[0], filters_list[1], 3),
            ]
        )
        m.add_module(
            "conv_out",
            nn.Conv2d(
                filters_list[1],
                out_filter,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.out0, x0)
        #  yolo branch 1
        x1_in = self.out1_cbl(out0_branch)
        x1_in = self.out1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.out1, x1_in)
        #  yolo branch 2
        x2_in = self.out2_cbl(out1_branch)
        x2_in = self.out2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.out2, x2_in)

        outputs = [out0, out1, out2]

        return outputs


def yolov3(pretrain_path=None, model_path=None):
    model = YOLOv3(pretrain_path)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {}
        for key, val in state_dict.items():
            key = key.replace("module.", "")
            new_state_dict[key] = val

        model.load_state_dict(new_state_dict)
    return model


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def generalized_batched_nms(
    boxes, scores, idxs, iou_threshold, score_threshold=0.001, nms_type="normal"
):
    assert boxes.shape[-1] == 4

    if nms_type == "normal":
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError('NMS type not implemented: "{}"'.format(nms_type))

    return keep


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    return area_i / (area_a[:, None] + area_b - area_i)


def decode_predictions(
    input, anchors, image_size, num_classes, num_anchors, is_train=False
):
    bs = input.size(0)  # batch_size
    in_h = input.size(2)  # input_height
    in_w = input.size(3)  # input_weight
    stride_h = image_size[1] / in_h
    stride_w = image_size[0] / in_w
    bbox_attrs = 1 + 4 + num_classes
    prediction = (
        input.view(bs, num_anchors, bbox_attrs, in_h, in_w)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )

    # Get outputs
    scaled_anchors = [(a_w, a_h) for a_w, a_h in anchors]
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    # conf = torch.sigmoid(prediction[..., 4])  # Conf
    # pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
    pred_confs = prediction[..., 4]  # Conf
    pred_cls = prediction[..., 5:]  # Cls pred.

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # Calculate offsets for each grid
    grid_x = (
        torch.linspace(0, in_w - 1, in_w)
        .repeat(in_h, 1)
        .repeat(bs * num_anchors, 1, 1)
        .view(x.shape)
        .type(FloatTensor)
    )
    grid_y = (
        torch.linspace(0, in_h - 1, in_h)
        .repeat(in_w, 1)
        .t()
        .repeat(bs * num_anchors, 1, 1)
        .view(y.shape)
        .type(FloatTensor)
    )
    # Calculate anchor w, h
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
    # Add offset and scale with anchors
    pred_boxes = prediction[..., :4].clone()
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    pred_boxes[..., 0] *= stride_w
    pred_boxes[..., 1] *= stride_h
    pred_boxes = pred_boxes.data
    if is_train:
        return pred_confs, x, y, w, h, pred_cls, pred_boxes
    else:
        pred_confs = torch.sigmoid(pred_confs)
        pred_cls = torch.sigmoid(pred_cls)

        pred_boxes = pred_boxes.detach()
        pred_confs = pred_confs.detach()
        pred_cls = pred_cls.detach()

        output = torch.cat(
            (
                pred_boxes.view(bs, -1, 4),
                pred_confs.view(bs, -1, 1),
                pred_cls.view(bs, -1, num_classes),
            ),
            -1,
        )
        return output


def decode_outputs(
    outputs,
    image_size,
    num_classes,
    anchors=[
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [42, 119]],
        [[10, 13], [16, 30], [33, 23]],
    ],
):
    predictions = [
        decode_predictions(
            out,
            a,
            image_size,
            num_classes,
            len(a),
            is_train=False,
        )
        for out, a in zip(outputs, anchors)
    ]
    predictions = torch.cat(predictions, 1)
    return predictions


def postprocess(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.5, nms_type="normal"
):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        confidence = detections[:, 4] * detections[:, 5]
        nms_out_index = generalized_batched_nms(
            detections[:, :4],
            confidence,
            detections[:, -1],
            nms_thre,
            nms_type=nms_type,
        )
        detections[:, 4] = confidence / detections[:, 5]

        detections = detections[nms_out_index]

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].unique()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))
    return output


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

    parser = argparse.ArgumentParser(description="Yolov3")
    parser.add_argument("--pretrain-path", default=None, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    args = parser.parse_args()
    model = yolov3(args.pretrain_path, args.model_path)
    print(model)
