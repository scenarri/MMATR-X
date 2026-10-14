_base_ = [
    '../_base_/models/upernet_beit.py',
    '../_custom_dataset_/whu20_rgb_sar.py',
    '../_base_/default_runtime.py',
]

rgb_pretrained = '/root/autodl-tmp/weights/INP_dec8_RGB/checkpoint-99.pth'
sar_pretrained = '/root/autodl-tmp/weights/INP_dec8_SAR/checkpoint-99.pth'

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)

itpn_common = dict(
    type='iTPN_pixel_dualembed',
    img_size=256,
    patch_size=16,
    embed_dim=512,
    mlp_depth1=3,
    mlp_depth2=3,
    depth=24,
    num_heads=8,
    mlp_ratio=4,
    fpn_dim=256,
    fpn_depth=1,
    qkv_bias=True,
    ape=True,
    rpe=False,
    drop_path_rate=0.1,
    num_outs=4,
    use_checkpoint=True,
    frozen=False)

model = dict(
    type='DualModalEncoderDecoder',
    rgb_channels=3,
    sar_channels=3,
    data_preprocessor=dict(
        _delete_=True,
        type='DualModalSegDataPreProcessor',
        rgb_mean=[41.7901, 38.9267, 32.108],
        rgb_std=[18.4388, 18.8588, 17.9144],
        sar_mean=[54.0325, 54.0325, 54.0325],
        sar_std=[48.7241, 48.7241, 48.7241],
        modality_channels=(3, 3),
        bgr_to_rgb=(True, True),
        size=(256, 256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='IndependentDualModalITPN',
        rgb_backbone=dict(
            **itpn_common,
            modality='RGB',
            init_cfg=dict(type='Pretrained', checkpoint=rgb_pretrained)),
        sar_backbone=dict(
            **itpn_common,
            modality='SAR',
            init_cfg=dict(type='Pretrained', checkpoint=sar_pretrained))),
    neck=None,
    decode_head=dict(
        in_channels=[512, 512, 512, 512],
        channels=768,
        num_classes=7),
    auxiliary_head=dict(in_channels=512, num_classes=7),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='DualModalITPNLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_type='layer_wise', num_layers=30, decay_rate=0.93),
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=7200, val_interval=800)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=300,
        end=7200,
        by_epoch=False,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=None,
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
