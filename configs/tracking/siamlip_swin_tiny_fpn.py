_base_ = [
    './antiuav410_tracking.py',
    '../../libs/mmdetection-main/configs/_base_/schedules/schedule_1x.py', 
    '../../libs/mmdetection-main/configs/_base_/default_runtime.py'
]
import sys
sys.path.insert(0, 'libs/mmdetection-main')

load_from = 'libs/mmdetection-main/checkpoints/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410-ba97be24.pth'
lang_model_name = 'libs/bert-base-uncased'  


motion_history_max_len = 5
model = dict(
    type='GLIPTracking',
    data_preprocessor=dict(
        type='SiameseDetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],   #bgr
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False),
    neck=dict(
        type='FPN_DropBlock',
        in_channels=[192, 384, 768], 
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],  # 对齐 P2-P7 的步长
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        early_fuse=True,
        use_checkpoint=True,
        ),
    language_model=dict(type='BertModel', 
    name=lang_model_name),  


    fusion_module=dict(
        type='MFDCAFusion',
        in_channels=[256, 256, 256, 256,256],
        norm_cfg=dict(type='BN', requires_grad=True)
        ),

    motion_history_max_len=motion_history_max_len,
    iou_thr=0.3,
  
    train_cfg=dict(
        assigner=dict(
            type='CustomATSSAssigner',
            topk=9,
            iou_calculator=dict(type='BboxOverlaps2D_GLIP')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.00,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))



dataset_type = 'AntiUAV410TrackingDataset'
# data_root = '/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/'
data_root = '/media/data2/TrackingDatasets/Anti-UAVALLFinal/'

train_pipeline = [
    dict(
        type='SiameseLoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=None),
    dict(type='SiameseLoadAnnotations', with_bbox=True),
    dict(type='SiameseGTBoxSubOne_GLIP'),
    dict(
        type='SiameseRandomChoiceResize',
        scales=[(1333, 480), (1333, 560), (1333, 640), (1333, 720),
                (1333, 800)],
        keep_ratio=True,
        resize_type='SiameseFixScaleResize',
        backend='pillow'),
    dict(type='SiameseRandomFlip_GLIP', prob=0.5),
    dict(type='SiameseFilterAnnotations', min_gt_bbox_wh=(1, 1)),

    dict(
        type='SiameseBuildMotionHistory',
        history_key='motion_history_gt',
        fmt='xyxy',
        motion_history_max_len=motion_history_max_len),
    dict(
        type='SiamesePackDetInputs',
        meta_keys=('img_id', 'T_img_path', 'S_img_path', 'T_ori_shape', 'S_ori_shape', 'T_img_shape', 'S_img_shape',
                   'T_scale_factor', 'S_scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'),
        motion_history_max_len=motion_history_max_len)
]


test_pipeline = [
    dict(
        type='SiameseLoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='SiameseFixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='SiameseLoadAnnotations', with_bbox=True),
    dict(
        type='SiamesePackDetInputs',
        meta_keys=('img_id', 'T_img_path', 'S_img_path', 'T_ori_shape', 'S_ori_shape', 'T_img_shape', 'S_img_shape',
                   'T_scale_factor', 'S_scale_factor', 'text', 'custom_entities'),
        motion_history_max_len=motion_history_max_len)
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            # ann_file='/media/data1/wangjiaqi/Code/SiamGLIP_fortracking/annotationsantiuav_410/train.json',
            ann_file='/media/data1/wangjiaqi/Code/SiamGLIP_fortracking/annotationsantiuav_Fianl/train.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            motion_history_max_len=motion_history_max_len,
            return_classes=True,
            backend_args=None)))

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        return_classes=True,
        motion_history_max_len=motion_history_max_len))
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=8, type='EpochBasedTrainLoop', val_interval=16)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=1, norm_type=2))

custom_hooks = [
    dict(
        type='FusionAlphaWarmupHook',
        alpha_max=0.1,
        warmup_epochs=4,
        start_epoch=2,
        module_attr='fusion_module')
]
