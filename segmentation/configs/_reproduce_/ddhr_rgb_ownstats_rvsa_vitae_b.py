_base_ = [
    '../_base_/models/upernet_beit.py',
    '../_custom_dataset_/ddhrsk_rgb.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.rvsa_vitae',
        'mmseg.engine.optimizers.rvsa_layer_decay',
    ],
    allow_failed_imports=False)

pretrained = (
    'reproduce_weights/RVSA/'
    'vitae-b-checkpoint-1599-transform-no-average.pth')

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[162.457, 164.329, 166.7645],
        std=[41.7551, 39.3668, 37.239],
        bgr_to_rgb=True,
        size=(256, 256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='ViTAE_NC_Win_RVSA_V3_WSZ7',
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        out_indices=(3, 5, 7, 11),
        interval=3,
        use_abs_pos_emb=True,
        # Reuse the 14x14 MAE sin-cos embedding at the 16x16 patch grid.
        # Set False only when reproducing the old loader's mismatch-and-skip
        # behavior exactly.
        interpolate_pos_embed=True,
        # The official repository recommends SyncBN for multi-GPU ViTAE.
        pcm_norm='SyncBN',
        # PCM contains BatchNorm, so checkpointing would update its running
        # statistics again during backward recomputation.
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=None,
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=768,
        num_classes=5,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        in_channels=768,
        in_index=2,
        channels=256,
        num_classes=5,
        norm_cfg=norm_cfg),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ``backbone.norm`` is retained for checkpoint compatibility but is not used
# by the official dense-prediction forward, hence DDP must track unused params.
find_unused_parameters = True

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='RVSALayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_type='layer_wise', num_layers=12, decay_rate=0.75),
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
