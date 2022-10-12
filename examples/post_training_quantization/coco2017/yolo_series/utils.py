import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat


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


def decode_outputs(
    outputs,
    arch,
    image_size,
    num_classes,
    conf_thre=0.7,
    nms_thre=0.5,
):
    anchors_dict = {
        "yolov3": [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [42, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ],
        "yolov4": [
            [[142, 110], [192, 243], [459, 401]],
            [[36, 75], [76, 55], [72, 146]],
            [[12, 16], [19, 36], [40, 28]],
        ],
    }
    anchors = anchors_dict[arch]
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
    detections = postprocess(predictions, num_classes, conf_thre, nms_thre)
    return detections
