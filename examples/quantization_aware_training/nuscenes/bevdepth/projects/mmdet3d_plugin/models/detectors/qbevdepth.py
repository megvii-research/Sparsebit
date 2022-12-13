import copy
import os
import torch
import torch.nn as nn
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result
from mmdet.core import multi_apply


class SELikeModule(nn.Module):
    def __init__(self, module):
        super(SELikeModule, self).__init__()
        self.module = module
        self.feat_quant = nn.Identity()
        self.feat_quant.remove = False  # a hack impl of quant the input of LSS
        self.attn_quant = nn.Identity()
        self.attn_quant.remove = False  # a hack impl of quant the input of LSS

    def forward(self, x, cam_params):
        x = self.module.input_conv(x)
        b, c, _, _ = x.shape
        y = self.module.fc(cam_params).view(b, c, 1, 1)
        x = self.feat_quant(x)
        y = self.attn_quant(y.expand_as(x))
        return x * y


class ViewTransformerLSSBEVDepthForward(nn.Module):
    def __init__(self, model):
        super(ViewTransformerLSSBEVDepthForward, self).__init__()
        self.model = model

    def forward(self, input, img_feat, depth_digit):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape

        depth_prob = self.model.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.model.numC_Trans, self.model.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.model.accelerate:
            bev_feat = self.model.voxel_pooling_accelerated(
                rots, trans, intrins, post_rots, post_trans, volume
            )
        else:
            geom = self.model.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.model.voxel_pooling(geom, volume)
        return bev_feat


class BEVDepthTraced(nn.Module):
    def __init__(self, model):
        super(BEVDepthTraced, self).__init__()
        _model = copy.deepcopy(model)
        self.img_backbone = _model.img_backbone
        self.img_neck = _model.img_neck
        self.img_view_transformer_featnet = _model.img_view_transformer.featnet
        self.se = SELikeModule(_model.img_view_transformer.se)
        self.extra_depthnet = _model.img_view_transformer.extra_depthnet
        self.dcn = _model.img_view_transformer.dcn
        self.depthnet = _model.img_view_transformer.depthnet
        _model.img_view_transformer.featnet = nn.Identity()
        _model.img_view_transformer.se = nn.Identity()
        _model.img_view_transformer.extra_depthnet = nn.Identity()
        _model.img_view_transformer.dcn = nn.Identity()
        _model.img_view_transformer.depthnet = nn.Identity()
        self.img_view_transformer = ViewTransformerLSSBEVDepthForward(
            _model.img_view_transformer
        )
        self.bev_encoder_backbone = _model.img_bev_encoder_backbone
        self.bev_encoder_neck = _model.img_bev_encoder_neck
        self.head = _model.pts_bbox_head
        self.head_shared_conv = _model.pts_bbox_head.shared_conv
        self.head_task_heads = _model.pts_bbox_head.task_heads
        self.img_view_transformer_featnet_quant = nn.Identity()
        self.img_view_transformer_featnet_quant.remove = (
            False  # a hack impl of quant the input of LSS
        )
        self.img_view_transformer_depth_quant = nn.Identity()
        self.img_view_transformer_depth_quant.remove = (
            False  # a hack impl of quant the input of LSS
        )
        self.loss = _model.pts_bbox_head.loss
        self.get_bboxes = _model.pts_bbox_head.get_bboxes

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_view_transformer_encoder(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, oldC, H, W = x.shape  # 512
        x = x.view(B * N, oldC, H, W)
        img_feat = self.img_view_transformer_featnet(x)
        img_feat = self.img_view_transformer_featnet_quant(img_feat)

        depth_feat = x
        cam_params = torch.cat(
            [
                intrins.reshape(B * N, -1),
                post_rots.reshape(B * N, -1),
                post_trans.reshape(B * N, -1),
                rots.reshape(B * N, -1),
                trans.reshape(B * N, -1),
            ],
            dim=1,
        )
        depth_feat = self.se(depth_feat, cam_params)
        depth_feat = self.extra_depthnet(depth_feat)[0]
        depth_feat = self.dcn(depth_feat)
        depth_digit = self.depthnet(depth_feat)
        depth_digit = self.img_view_transformer_depth_quant(depth_digit)

        return img_feat, depth_digit

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

    def forward(self, img_inputs, img_metas=None, rescale=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        img, rots, trans, intrins, post_rots, post_trans, depth_gt = img_inputs
        x = self.image_encoder(img)
        img_feat, depth_digit = self.image_view_transformer_encoder(
            [x, rots, trans, intrins, post_rots, post_trans]
        )
        x = self.img_view_transformer(
            [x, rots, trans, intrins, post_rots, post_trans], img_feat, depth_digit
        )
        x = self.bev_encoder(x)
        x = self.forward_pts_head([x])
        return x


class BEVDepthForward(Base3DDetector):
    def __init__(self, ori_model, graph_module):
        super(BEVDepthForward, self).__init__()
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
        bbox_list = [dict() for _ in range(len(img_metas))]
        outs = self.graph_module(img)
        bbox_list = self.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
