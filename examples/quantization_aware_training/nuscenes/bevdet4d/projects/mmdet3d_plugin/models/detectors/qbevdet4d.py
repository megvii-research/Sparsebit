import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result
from mmdet.core import multi_apply
from mmdet.models import DETECTORS


class ShiftFeature(nn.Module):
    def __init__(self, bx, dx, interpolation_mode):
        super(ShiftFeature, self).__init__()
        self.bx = bx
        self.dx = dx
        self.interpolation_mode = interpolation_mode

    def forward(self, input, trans, rots):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = (
            torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device)
            .view(1, w)
            .expand(h, w)
        )
        ys = (
            torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device)
            .view(h, 1)
            .expand(h, w)
        )
        grid = (
            torch.stack((xs, ys, torch.ones_like(xs)), -1)
            .view(1, h, w, 3)
            .expand(n, h, w, 3)
            .view(n, h, w, 3, 1)
        )
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
        l02c[:, :, :3, :3] = rots[0]
        l02c[:, :, :3, 3] = trans[0]
        l02c[:, :, 3, 3] = 1

        l12c = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
        l12c[:, :, :3, :3] = rots[1]
        l12c[:, :, :3, 3] = trans[1]
        l12c[:, :, 3, 3] = 1
        # l0tol1 = l12c.matmul(torch.inverse(l02c))[:,0,:,:].view(n,1,1,4,4)
        l0tol1 = l02c.matmul(torch.inverse(l12c))[:, 0, :, :].view(n, 1, 1, 4, 4)

        l0tol1 = l0tol1[:, :, :, [True, True, False, True], :][
            :, :, :, :, [True, True, False, True]
        ]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.dx[0]
        feat2bev[1, 1] = self.dx[1]
        feat2bev[0, 2] = self.bx[0] - self.dx[0] / 2.0
        feat2bev[1, 2] = self.bx[1] - self.dx[1] / 2.0
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l0tol1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor(
            [w - 1.0, h - 1.0], dtype=input.dtype, device=input.device
        )
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(
            input,
            grid.to(input.dtype),
            align_corners=True,
            mode=self.interpolation_mode,
        )
        return output


@DETECTORS.register_module()
class BEVDet4dTraced(nn.Module):
    def __init__(self, model, batch_size):
        super(BEVDet4dTraced, self).__init__()
        _model = copy.deepcopy(model)
        self.batch_size = batch_size
        self.detach = _model.detach
        self.aligned = _model.aligned
        self.before = _model.before
        self.img_backbone = _model.img_backbone
        self.img_neck = _model.img_neck
        self.img_view_transformer_depthnet = _model.img_view_transformer.depthnet
        _model.img_view_transformer.depthnet = nn.Identity()
        self.img_view_transformer = _model.img_view_transformer
        self.pre_process = _model.pre_process
        self.pre_process_net = _model.pre_process_net
        self.shift_feature = ShiftFeature(
            _model.img_view_transformer.bx,
            _model.img_view_transformer.dx,
            _model.interpolation_mode,
        )
        self.bev_encoder_backbone = _model.img_bev_encoder_backbone
        self.bev_encoder_neck = _model.img_bev_encoder_neck
        self.head = _model.pts_bbox_head
        self.head_shared_conv = _model.pts_bbox_head.shared_conv
        self.head_task_heads = _model.pts_bbox_head.task_heads
        self.img_view_transformer_quant = nn.Identity()
        self.img_view_transformer_quant.remove = (
            False  # a hack impl of quant the input of LSS
        )
        self.loss = _model.pts_bbox_head.loss
        self.get_bboxes = _model.pts_bbox_head.get_bboxes

    def image_encoder(self, imgs):
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.reshape(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_view_transformer_encoder(self, x):
        B, num_cams, oldC, H, W = x.shape  # 512
        x = x.view(B * num_cams, oldC, H, W)
        x = self.img_view_transformer_depthnet(x)
        x = self.img_view_transformer_quant(x)
        x = x.view(B, num_cams, -1, H, W)
        return x

    def bev_encoder(self, x):
        x = self.bev_encoder_backbone(x)
        x = self.bev_encoder_neck(x)
        return x

    def forward_single(self, x):
        x = self.head_shared_conv(x)
        task_heads = self.head_task_heads
        # unfold loop
        ret_dicts = []
        ret_dicts.append(task_heads[0](x))
        ret_dicts.append(task_heads[1](x))
        ret_dicts.append(task_heads[2](x))
        ret_dicts.append(task_heads[3](x))
        ret_dicts.append(task_heads[4](x))
        ret_dicts.append(task_heads[5](x))
        return ret_dicts

    def forward_pts_head(self, feats):
        return multi_apply(self.forward_single, feats)

    def enable_traced(self):
        self.traced = True

    def forward(self, inputs, img_metas=None, rescale=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        imgs, rots, trans, intrins, post_rots, post_trans = inputs
        B, _, _, _, _ = imgs.shape
        x = self.image_encoder(imgs)
        x = self.image_view_transformer_encoder(x)
        x1 = self.img_view_transformer(
            [
                x[:, 0:12:2].contiguous(),
                rots[:, :6],
                trans[:, :6],
                intrins[:, :6],
                post_rots[:, :6],
                post_trans[:, :6],
            ]
        )
        x2 = self.img_view_transformer(
            [
                x[:, 1:12:2].contiguous(),
                rots[:, :6],
                trans[:, :6],
                intrins[:, 6:],
                post_rots[:, 6:],
                post_trans[:, 6:],
            ]
        )
        x = torch.cat((x1, x2), dim=0)
        if self.before and self.pre_process:
            x = self.pre_process_net(x)[0]
            x1, x2 = torch.split(x, B, dim=0)
        x2 = self.shift_feature(
            x2, [trans[:, :6], trans[:, 6:]], [rots[:, :6], rots[:, 6:]]
        )
        if self.pre_process and not self.before:
            x1 = self.pre_process_net(x1)[0]
            x2 = self.pre_process_net(x2)[0]
        if self.detach:
            x2 = x2.detach()
        x = torch.cat([x1, x2], dim=1)
        x = self.bev_encoder(x)
        x = self.forward_pts_head([x])
        return x


@DETECTORS.register_module()
class BEVDet4dForward(Base3DDetector):
    def __init__(self, ori_model, graph_module):
        super(BEVDet4dForward, self).__init__()
        self.graph_module = graph_module
        self.loss = ori_model.loss
        self.get_bboxes = ori_model.get_bboxes

    def extract_feat(self):
        pass

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img_inputs=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        outs = self.graph_module(img_inputs)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.loss(*loss_inputs)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, "img_inputs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(img_inputs), len(img_metas)
                )
            )

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get("combine_type", "output")
        if combine_type == "output":
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type == "feature":
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        outs = self.graph_module(img)
        bbox_list = self.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
