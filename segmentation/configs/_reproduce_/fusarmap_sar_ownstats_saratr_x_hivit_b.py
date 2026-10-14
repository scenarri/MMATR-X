_base_ = [
    '../_base_/models/upernet_mae.py',
    '../_custom_dataset_/fusarmap.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.saratr_x_hivit',
        'mmseg.engine.optimizers.saratr_x_layer_decay',
    ],
    allow_failed_imports=False)

pretrained = (
    'reproduce_weights/SARATR-X/186K_notest_checkpoint-600.pth')

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[59.0409, 59.0409, 59.0409],
        std=[53.4691, 53.4691, 53.4691],
        bgr_to_rgb=True,
        size=(512, 512),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SARATRXHiViT',
        # 224 defines the checkpoint's 14x14 absolute-position embedding;
        # arbitrary inputs divisible by 16 are interpolated at runtime.
        img_size=224,
        patch_size=16,
        inner_patches=4,
        in_channels=3,
        embed_dims=512,
        depths=(2, 2, 20),
        num_heads=8,
        stem_mlp_ratio=3.,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ape=True,
        rpe=False,
        patch_norm=True,
        out_indices=('H', 'M', 19, 19),
        global_indices=(4, 9, 14, 19),
        window_size=14,
        with_cp=True,
        expand_single_channel=True,
        strict_encoder_load=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    # HiViT already exposes /4, /8, /16 and /32 maps. UPerHead performs
    # lateral projection and top-down pyramid fusion itself.
    neck=None,
    decode_head=dict(
        in_channels=[128, 256, 512, 512],
        in_index=[0, 1, 2, 3],
        channels=768,
        num_classes=5,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        in_channels=512,
        in_index=2,
        channels=256,
        num_classes=5,
        norm_cfg=norm_cfg),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(256, 256), crop_size=(512, 512)))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='SARATRXLayerDecayOptimizerConstructor',
    # 4 + 4 + 20 residual blocks; the constructor additionally counts the
    # two trainable PatchMerge transitions, matching the iTPN convention.
    paramwise_cfg=dict(num_layers=28, decay_rate=0.93),
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
