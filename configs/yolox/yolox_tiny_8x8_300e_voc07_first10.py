_base_ = './yolox_s_8x8_300e_voc07.py'

# model settings
model = dict(
    random_size_range=(10, 20),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96, num_classes=10))

img_scale = (640, 640)  # height, width

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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

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
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 
            #    'diningtable', 'dog', 'horse',
            #    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            #    'tvmonitor'
               )

data_root = 'data/my_voc/'

train_dataset = dict(pipeline=train_pipeline,
                    dataset=dict(
                            ann_file=data_root + 'annotations/voc07_trainval_sel_first_10_cats.json',
                            classes=CLASSES,
                        ),
)

data = dict(
    train=train_dataset,
    val=dict(pipeline=test_pipeline,
            ann_file=data_root + 'annotations/voc07_test_sel_first_10_cats.json',
            classes=CLASSES),
    test=dict(pipeline=test_pipeline,
            ann_file=data_root + 'annotations/voc07_test_sel_first_10_cats.json',
            classes=CLASSES))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
checkpoint_config = dict(interval=50)
# checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=10,
    )
log_config = dict(interval=20)