_base_ = [
    '../_base_/models/upernet_beit.py',
    '../_custom_dataset_/ddhrsk_sar.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_80k.py'
]

pretrained = '/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth'

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[50.774, 50.774, 50.774],
        std=[49.6096, 49.6096, 49.6096],
        bgr_to_rgb=True,
        size=(256,256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        type='iTPN_pixel_dualembed',
        modality='SAR',
        frozen=False,
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=None,
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        num_classes=5,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=5
    ),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='itpn_LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_type': 'layer_wise',
        'num_layers': 30,
        'decay_rate': 0.93,
        'absolute_pos_embed': dict(decay_mult=0.),
        'relative_position_bias_table': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    },
    optimizer=dict(
        type='AdamW',
        lr=3e-05,
        betas=(0.9, 0.999),
        weight_decay=0.05))


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