classes = ['crown']
num_classes=len(classes)

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            max_per_img=300,
            mask_thr_binary=0.5)))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        #  img_scale=[(800, 800), (1024, 1024), (1280, 1280)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset1 = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Dataset_1_train/annotation/Dataset_1_train.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Dataset_1_train/images',
    pipeline=train_pipeline)

dataset2 = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Dataset_2_train/annotation/Dataset_2_train.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Dataset_2_train/images',
    pipeline=train_pipeline)

dataset3 = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Dataset_3_train/annotation/Dataset_3_train.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Dataset_3_train/images',
    pipeline=train_pipeline)

dataset4 = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Dataset_4_train/annotation/Dataset_4_train.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Dataset_4_train/images',
    pipeline=train_pipeline)

dataset5 = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Dataset_5_train/annotation/Dataset_5_train.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Dataset_5_train/images',
    pipeline=train_pipeline)

dataset_val = dict(
    type='CocoDataset',
    classes=classes,
    ann_file='/home/user/data/rsipac2023/guoji/train/Validation_set/image_ids/validation_img_id.json',
    img_prefix='/home/user/data/rsipac2023/guoji/train/Validation_set/images',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='ConcatDataset',
            datasets=[dataset1, dataset2, dataset3, dataset4, dataset5]
        )),
    val=dataset_val,
    test=dataset_val)
evaluation = dict(interval=2, metric=['bbox', 'segm'], start=1)
optimizer = dict(type='SGD', lr=0.0025*1, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=12, save_optimizer=False)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    #  warmup='linear',
    #  warmup_iters=500,
    #  warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
#  load_from = 'weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
fp16 = dict(loss_scale=512.)
