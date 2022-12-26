# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = [
    "../../configs/_base_/datasets/nus-3d.py",
    "../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

data_config = {
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.04,
}

# Model
grid_config = {
    "xbound": [-51.2, 51.2, 0.8],
    "ybound": [-51.2, 51.2, 0.8],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64


model = dict(
    type="BEVDetSequentialES",
    aligned=True,
    detach=True,
    before=True,
    img_backbone=dict(
        pretrained="torchvision://resnet50",
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style="pytorch",
    ),
    img_neck=dict(
        type="FPNForBEVDet",
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    img_view_transformer=dict(
        type="ViewTransformerLiftSplatShoot",
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
    ),
    img_bev_encoder_backbone=dict(
        type="ResNetForBEVDet", numC_input=128, num_channels=[128, 256, 512]
    ),
    img_bev_encoder_neck=dict(
        type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256
    ),
    pre_process=dict(
        type="ResNetForBEVDet",
        numC_input=numC_Trans,
        num_layer=[
            2,
        ],
        num_channels=[
            64,
        ],
        stride=[
            1,
        ],
        backbone_output_ids=[
            0,
        ],
    ),
    pts_bbox_head=dict(
        type="CenterHead",
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2
            # Scale-NMS
            nms_type=["rotate", "rotate", "rotate", "circle", "rotate", "rotate"],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0,
                [0.7, 0.7],
                [0.4, 0.55],
                1.1,
                [1.0, 1.0],
                [4.5, 9.0],
            ],
        )
    ),
)


# Data
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")


train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_BEVDet",
        is_train=True,
        data_config=data_config,
        sequential=True,
        aligned=True,
        trans_only=False,
    ),
    dict(
        type="LoadPointsFromFile",
        dummy=True,
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True,
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["img_inputs", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "img_info",
        ),
    ),
]

test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_BEVDet",
        data_config=data_config,
        sequential=True,
        aligned=True,
        trans_only=False,
    ),
    # load lidar points for --show in test.py only
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="Collect3D",
                keys=["points", "img_inputs"],
                meta_keys=(
                    "filename",
                    "ori_shape",
                    "img_shape",
                    "lidar2img",
                    "depth2img",
                    "cam2img",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "pcd_horizontal_flip",
                    "pcd_vertical_flip",
                    "box_mode_3d",
                    "box_type_3d",
                    "img_norm_cfg",
                    "pcd_trans",
                    "sample_idx",
                    "pcd_scale_factor",
                    "pcd_rotation",
                    "pts_filename",
                    "transformation_3d_flow",
                    "adjacent",
                    "adjacent_type",
                ),
            ),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_BEVDet",
        data_config=data_config,
        sequential=True,
        aligned=True,
        trans_only=False,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["img_inputs"]),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "nuscenes_infos_train_4d_interval3_max60.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
            img_info_prototype="bevdet_sequential",
            speed_mode="abs_dis",
            max_interval=9,
            min_interval=2,
            prev_only=True,
            fix_direction=True,
        ),
    ),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + "nuscenes_infos_val_4d_interval3_max60.pkl",
        modality=input_modality,
        img_info_prototype="bevdet_sequential",
    ),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + "nuscenes_infos_val_4d_interval3_max60.pkl",
        modality=input_modality,
        img_info_prototype="bevdet_sequential",
        max_interval=10,
        fix_direction=True,
    ),
)

# Optimizer
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[7, 10]
)
runner = dict(type="EpochBasedRunner", max_epochs=12)

"""
checkpoint:
float mAP: 0.322
float NDS: 0.457
**************************** 4w4f *************************
qconfig: qconfig_r50_lsq_4w4f.yaml
mAP: 0.3169                                                                                                                                                                            
mATE: 0.6947                                                                                                                                                                           
mASE: 0.2787                                                                                                                                                                           
mAOE: 0.4954                                                                                                                                                                           
mAVE: 0.3429                                                                                                                                                                           
mAAE: 0.1941                                                                                                                                                                           
NDS: 0.4579                                                                                                                                                                            
Eval time: 128.2s                                                                                                                                                                                                                                                                                                                                                             Per-class results:                                                                                                                                                                     Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                                                                                            car     0.518   0.530   0.164   0.115   0.363   0.204                                                                                                                                 
truck   0.244   0.681   0.221   0.125   0.308   0.212                                                                                                                                  
bus     0.316   0.711   0.197   0.071   0.600   0.243                                                                                                                                  
trailer 0.145   1.046   0.253   0.488   0.205   0.053                                                                                                                                  
construction_vehicle    0.069   0.894   0.511   1.046   0.130   0.369                                                                                                                  
pedestrian      0.355   0.757   0.299   0.812   0.518   0.249                                                                                                                          
motorcycle      0.277   0.694   0.254   0.707   0.451   0.212                                                                                                                          
bicycle 0.248   0.583   0.264   0.979   0.166   0.012                                                                                                                                  
traffic_cone    0.493   0.553   0.342   nan     nan     nan                                                                                                                            
barrier 0.504   0.497   0.281   0.115   nan     nan
"""
