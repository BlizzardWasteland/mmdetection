checkpoint_config = dict(interval=50)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='SyncNormHook', num_last_epochs=15, interval=50, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
custom_imports = dict(
    imports=['mmdet_custom.datasets', 'mmcv_custom.runner'],
    allow_failed_imports=False)
img_scale = (640, 640)
act_type = 'ReLU'
default_widen_factor = 0.375
default_deepen_factor = 0.33
widen_factor_range = [
    0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666,
    0.8333333333333334, 1
]
deepen_factor_range = [0, 0.3333333333333333, 0.6666666666666666, 1]
search_space = dict(
    backbone_widen_factor_range=[
        0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666,
        0.8333333333333334, 1
    ],
    backbone_deepen_factor_range=[
        0, 0.3333333333333333, 0.6666666666666666, 1
    ],
    neck_widen_factor_range=[1],
    head_widen_factor_range=[
        0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666,
        0.8333333333333334, 1
    ])
result = [4, 5, 5, 4, 0, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4]
arch = dict(
    widen_factor_backbone_idx=[4, 5, 5, 4, 0],
    deepen_factor_backbone_idx=[3, 3, 3, 2],
    widen_factor_neck_idx=[0, 0, 0, 0, 0, 0, 0, 0],
    widen_factor_neck_out_idx=4)
widen_factor_backbone = [0.3125, 0.375, 0.375, 0.3125, 0.0625]
deepen_factor_backbone = [0.33, 0.33, 0.33, 0.22]
widen_factor_neck = [1, 1, 1, 1, 1, 1, 1, 1]
widen_factor_head = [0.8333333333333334]
in_channels = [96, 160, 64]
head_channels = 80
model = dict(
    type='SearchableYOLOX_KD',
    is_kd=False,
    bn_training_mode=False,
    retraining=True,
    search_space=dict(
        backbone_widen_factor_range=[
            0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666,
            0.8333333333333334, 1
        ],
        backbone_deepen_factor_range=[
            0, 0.3333333333333333, 0.6666666666666666, 1
        ],
        neck_widen_factor_range=[1],
        head_widen_factor_range=[
            0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666,
            0.8333333333333334, 1
        ]),
    divisor=4,
    input_size=(640, 640),
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=dict(
        type='SearchableCSPDarknetWOSPP',
        deepen_factor=[0.33, 0.33, 0.33, 0.22],
        widen_factor=[0.3125, 0.375, 0.375, 0.3125, 0.0625],
        act_cfg=dict(type='ReLU')),
    neck=dict(
        type='SearchableYOLOXPAFPN',
        in_channels=[96, 160, 64],
        out_channels=80,
        widen_factor=[1, 1, 1, 1, 1, 1, 1, 1],
        num_csp_blocks=1,
        act_cfg=dict(type='ReLU')),
    bbox_head=dict(
        type='SearchableYOLOXHead',
        num_classes=15,
        in_channels=80,
        feat_channels=80,
        act_cfg=dict(type='ReLU')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/my_voc/'
dataset_type = 'CocoDatasetContinual'
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person')
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-320, -320)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDatasetContinual',
        ann_file=
        'data/my_voc/annotations/voc07_trainval_sel_first_15_cats.json',
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person'),
        previous_num_classes=0),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.5, 1.5),
            border=(-320, -320)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='Normalize',
            mean=[0, 0, 0],
            std=[255.0, 255.0, 255.0],
            to_rgb=True),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDatasetContinual',
            ann_file=
            'data/my_voc/annotations/voc07_trainval_sel_first_15_cats.json',
            img_prefix='data/VOCdevkit/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                     'horse', 'motorbike', 'person'),
            previous_num_classes=0),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.5, 1.5),
                border=(-320, -320)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDatasetContinual',
        ann_file='data/my_voc/annotations/voc07_test_sel_first_15_cats.json',
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person'),
        previous_num_classes=0),
    test=dict(
        type='CocoDatasetContinual',
        ann_file='data/my_voc/annotations/voc07_test_sel_first_15_cats.json',
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person'),
        previous_num_classes=0))
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
max_epochs = 300
num_last_epochs = 15
interval = 50
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(
    save_best='auto', interval=10, dynamic_intervals=[(285, 1)], metric='mAP')
work_dir = './work_dirs/yolox_search_voc07_first15'
auto_resume = False
gpu_ids = [0]
