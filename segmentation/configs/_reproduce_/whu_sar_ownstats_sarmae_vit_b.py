_base_ = [
    '../_base_/models/upernet_mae.py',
    '../_custom_dataset_/whu_sar.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.sarmae_vit',
        'mmseg.engine.optimizers.sarmae_layer_decay',
    ],
    allow_failed_imports=False)

pretrained = 'reproduce_weights/SARMAE/SARMAE_vitb_checkpoint-last'

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[54.0325, 54.0325, 54.0325],
        std=[48.7241, 48.7241, 48.7241],
        bgr_to_rgb=True,
        size=(256, 256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SARMAEVisionTransformer',
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.,
        out_indices=(3, 5, 7, 11),
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        with_cp=False,
        # The official newer timm port enables fused SDPA when supported.
        use_sdpa=True,
        strict_encoder_load=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    # The backbone already builds the /4, /8, /16 and /32 Simple FPN.
    neck=None,
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=768,
        num_classes=7,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        in_channels=768,
        in_index=2,
        channels=256,
        num_classes=7,
        norm_cfg=norm_cfg),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='SARMAELayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, decay_rate=0.75),
    optimizer=dict(
        type='AdamW',
        lr=3e-5,
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
