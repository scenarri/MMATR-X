_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_custom_dataset_/whu_rgb.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.gfm_swin',
        'mmseg.engine.optimizers.swin_layer_decay',
    ],
    allow_failed_imports=False)

pretrained = 'reproduce_weights/GFM/gfm.pth'

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[41.7901, 38.9267, 32.108],
        std=[18.4388, 18.8588, 17.9144],
        bgr_to_rgb=True,
        size=(256, 256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='GFMSwinTransformer',
        gfm_branch='encoder',
        pretrain_img_size=192,
        in_channels=3,
        embed_dims=128,
        patch_size=4,
        window_size=6,
        mlp_ratio=4,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        # Follow the default Swin-B UPerNet fine-tuning recipe.
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        with_cp=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=None,
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=768,
        num_classes=7,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        in_channels=512,
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
    constructor='SwinLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, decay_rate=0.90),
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
