# Paired DDHRSK RGB-SAR dataset settings.
custom_imports = dict(
    imports=['projects.dual_modal_itpn'], allow_failed_imports=False)

dataset_type = 'PairedWHUDataset'
data_root = '/root/WHU/'
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 128),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # Both modalities receive independently sampled photometric distortion;
    # resize, crop and flip remain spatially synchronized.
    dict(
        type='PairedPhotoMetricDistortion',
        apply_to_rgb=True,
        apply_to_sar=True),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'img_path2', 'seg_map_path', 'ori_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'reduce_zero_label',
                   'modality_channels'))
]

test_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'img_path2', 'seg_map_path', 'ori_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'reduce_zero_label',
                   'modality_channels'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_images_train_20',
            img_path2='sar_images_train_20',
            seg_map_path='anns_train_20'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_images_val',
            img_path2='sar_images_val',
            seg_map_path='anns_val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
