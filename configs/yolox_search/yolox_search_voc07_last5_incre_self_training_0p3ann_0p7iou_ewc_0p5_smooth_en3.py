_base_ = ['../_base_/default_runtime.py']

custom_imports=dict(imports=['mmdet_custom.datasets', 'mmcv_custom.runner'], allow_failed_imports=False) 

img_scale = (640, 640)
act_type = 'ReLU'

# checkpoint_config = dict(type='CheckpointHook_nolog', interval=1)

default_widen_factor = 0.375
default_deepen_factor = 0.33
# [0.125, 0.25, 0.375, 0.5]
widen_factor_range = [1/6, 2/6, 3/6, 4/6, 5/6, 1]
# [0.11, 0.22, 0.33]
deepen_factor_range = [0, 1/3, 2/3, 1]
search_space = dict(
    backbone_widen_factor_range = widen_factor_range,
    backbone_deepen_factor_range = deepen_factor_range,
    neck_widen_factor_range = [1],
    head_widen_factor_range = widen_factor_range
)

# AP = (0.9286574125289917,),  2.1 M, 1.11 GLOPS
result = [4, 5, 5, 4, 0, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4]
arch={'widen_factor_backbone_idx': result[:5], 'deepen_factor_backbone_idx': result[5:9], 'widen_factor_neck_idx': result[9:17], 'widen_factor_neck_out_idx': result[17]}

widen_factor_backbone = [search_space['backbone_widen_factor_range'][i] for i in arch['widen_factor_backbone_idx']] 
deepen_factor_backbone = [search_space['backbone_deepen_factor_range'][i] for i in arch['deepen_factor_backbone_idx']] 
widen_factor_neck = [search_space['neck_widen_factor_range'][i] for i in arch['widen_factor_neck_idx']] 
widen_factor_head = [search_space['head_widen_factor_range'][arch['widen_factor_neck_out_idx']]]

widen_factor_backbone = [default_widen_factor*alpha for alpha in widen_factor_backbone]
deepen_factor_backbone = [default_deepen_factor*alpha for alpha in deepen_factor_backbone]
in_channels = [int(c*alpha) for c,alpha in zip([256, 512, 1024], widen_factor_backbone[-3:])]
head_channels = int(256*default_widen_factor*widen_factor_head[0])

previous_num_classes = 15
# model settings
model = dict(
    type='SearchableYOLOX_KD_Incre_ST_EWC',
    ori_num_classes=previous_num_classes,
    ori_checkpoint_file='work_dirs/yolox_search_voc07_first15/epoch_300.pth',
    ori_config_file='configs/yolox_search/yolox_search_voc07_first15.py',
    ori_weighted_file='work_dirs/yolox_search_voc07_first15_ewc_weight/ewc_weight.pth',
    lamda=0.5,
    anno_threshold=0.3,
    iou_threshold=0.7,
    smooth=True,
    beta=1e-3,
    is_kd=False,
    bn_training_mode=False,
    retraining=True,
    search_space=search_space,
    divisor=4,
    input_size=img_scale,
    random_size_range=(10, 20), 
    random_size_interval=10,
    backbone=dict(
        type='SearchableCSPDarknetWOSPP',
        deepen_factor=deepen_factor_backbone,
        widen_factor=widen_factor_backbone,
        act_cfg=dict(type=act_type)),
    neck=dict(
        type='SearchableYOLOXPAFPN',
        in_channels=in_channels,
        out_channels=head_channels,
        widen_factor=widen_factor_neck, 
        num_csp_blocks=1,
        act_cfg=dict(type=act_type)),
    bbox_head=dict(
        type='SearchableYOLOXHeadIncre',
        ori_num_classes=15,
        num_classes=20,
        in_channels=head_channels,
        feat_channels=head_channels,
        act_cfg=dict(type=act_type)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))


# dataset settings
data_root = 'data/my_voc/'
dataset_type = 'CocoDatasetContinual'
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_trainval_sel_last_5_cats.json',
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
        classes=CLASSES,
        previous_num_classes=previous_num_classes,
    ),
    pipeline=train_pipeline)

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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_test.json',
        img_prefix='data/VOCdevkit/',
        pipeline=test_pipeline,
        classes=CLASSES,
        previous_num_classes=0,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc07_test.json',
        img_prefix='data/VOCdevkit/',
        pipeline=test_pipeline,
        classes=CLASSES,
        previous_num_classes=0,))



# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None) # 10e-4 -> 10e-7
# optimizer_config = dict(grad_clip=dict(max_norm=1000, norm_type=2)) # 10e-4 -> 10e-7

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 50

# learning policy
lr_config = dict(
    # _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    # dict(
    #     type='YOLOXModeSwitchHook',
    #     num_last_epochs=num_last_epochs,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
# checkpoint_config = dict(interval=interval)
checkpoint_config = dict(interval=50)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    # interval=interval,
    interval=10,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP')
log_config = dict(interval=10)

auto_scale_lr = dict(base_batch_size=64)
