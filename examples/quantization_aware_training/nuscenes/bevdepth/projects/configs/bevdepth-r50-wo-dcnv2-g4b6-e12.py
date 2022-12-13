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
    type="BEVDepth",
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
        type="ViewTransformerLSSBEVDepthWithoutDCNv2",
        loss_depth_weight=100.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        extra_depth_net=dict(
            type="ResNetForBEVDet",
            numC_input=256,
            num_layer=[
                3,
            ],
            num_channels=[
                256,
            ],
            stride=[
                1,
            ],
        ),
    ),
    img_bev_encoder_backbone=dict(type="ResNetForBEVDet", numC_input=numC_Trans),
    img_bev_encoder_neck=dict(
        type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256
    ),
    pts_bbox_head=dict(
        type="CenterHead",
        task_specific=True,
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
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
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
            # nms_thr=0.2,
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
    ),
    dict(
        type="LoadPointsFromFile",
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
    dict(
        type="PointToMultiViewDepth",
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
    dict(type="LoadMultiViewImageFromFiles_BEVDet", data_config=data_config),
    # load lidar points for --show in test.py only
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="PointToMultiViewDepth",
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
            dict(type="Collect3D", keys=["points", "img_inputs"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_BEVDet", data_config=data_config),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="PointToMultiViewDepth",
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["img_inputs"]),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
            img_info_prototype="bevdet",
        ),
    ),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
    ),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
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
float mAP: 0.336
float NDS: 0.409

**************************** 8w8f *************************
qconfig: qconfig_r50_lsq_8w8f.yaml

mAP: 0.3352                                                                                                                                                                            
mATE: 0.6531                                                                                                                                                                           
mASE: 0.2731                                                                                                                                                                           
mAOE: 0.5399                                                                                                                                                                           
mAVE: 0.8655                                                                                                                                                                           
mAAE: 0.2322                                                                                                                                                                           
NDS: 0.4112                                                                                                                                                                            
Eval time: 191.4s                                                                                                                                                                                                                                                                                                                                                             
Per-class results:                                                                                                                                                                     
Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                                                                                            
car     0.535   0.522   0.158   0.116   0.935   0.219                                                                                                                                  
truck   0.259   0.667   0.216   0.114   0.813   0.220                                                                                                                                  
bus     0.377   0.668   0.188   0.084   1.436   0.379                                                                                                                                  
trailer 0.195   0.913   0.225   0.435   0.736   0.144                                                                                                                                  
construction_vehicle    0.078   0.856   0.493   0.951   0.139   0.388                                                                                                                  
pedestrian      0.358   0.721   0.302   1.351   0.908   0.427                                                                                                                          
motorcycle      0.287   0.695   0.265   0.869   1.519   0.062                                                                                                                          
bicycle 0.259   0.507   0.273   0.827   0.437   0.019                                                                                                                                  
traffic_cone    0.510   0.499   0.332   nan     nan     nan                                                                                                                            
barrier 0.495   0.483   0.279   0.112   nan     nan

**************************** 4w4f *************************
qconfig: qconfig_r50_lsq_4w4f.yaml

mAP: 0.3327                                                                                                                                                                            
mATE: 0.6638                                                                                                                                                                           
mASE: 0.2729                                                                                                                                                                           
mAOE: 0.5532                                                                                                                                                                           
mAVE: 0.8726                                                                                                                                                                           
mAAE: 0.2290                                                                                                                                                                           
NDS: 0.4072                                                                                                                                                                            
Eval time: 153.8s                                                                                                                                                                      
                                                                                                                                                                                       
Per-class results:                                                                                                                                                                     
Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                                                                                            
car     0.531   0.522   0.159   0.121   0.923   0.217                                                                                                                                  
truck   0.260   0.675   0.216   0.118   0.829   0.231                                                                                                                                  
bus     0.379   0.712   0.187   0.104   1.528   0.358                                                                                                                                  
trailer 0.187   0.889   0.233   0.404   0.821   0.133                                                                                                                                  
construction_vehicle    0.079   0.893   0.476   0.991   0.191   0.371                                                                                                                  
pedestrian      0.354   0.737   0.302   1.378   0.901   0.437                                                                                                                          
motorcycle      0.284   0.697   0.262   0.903   1.413   0.069                                                                                                                          
bicycle 0.259   0.497   0.278   0.839   0.375   0.016                                                                                                                                  
traffic_cone    0.498   0.522   0.336   nan     nan     nan                                                                                                                            
barrier 0.496   0.495   0.280   0.121   nan     nan
"""
