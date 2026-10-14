_base_ = [
    '../_base_/models/upernet_mae.py',
    '../_custom_dataset_/ddhrsk_sar.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.summit_vitdet',
        'mmseg.engine.optimizers.summit_layer_decay',
    ],
    allow_failed_imports=False)

pretrained = 'reproduce_weights/SUMMIT/SUMMIT.pth'

auto_scale_lr = dict(enable=True, base_batch_size=32)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[50.774, 50.774, 50.774],
        std=[49.6096, 49.6096, 49.6096],
        bgr_to_rgb=True,
        size=(256, 256),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SUMMITViTDet',
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        # ViTDet-B uses global attention at blocks 2, 5, 8 and 11.
        window_block_indices=(0, 1, 3, 4, 6, 7, 9, 10),
        # SUMMIT was pre-trained at 448: 28x28 patches plus one cls position.
        pretrain_img_size=448,
        pretrain_use_cls_token=True,
        with_cp=False,
        strict_encoder_load=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='SUMMITSimpleFPN',
        backbone_channel=768,
        in_channels=(192, 384, 768, 768),
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=768,
        num_classes=5,
        norm_cfg=norm_cfg),
    auxiliary_head=dict(
        in_channels=256,
        in_index=2,
        channels=256,
        num_classes=5,
        norm_cfg=norm_cfg),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='SUMMITLayerDecayOptimizerConstructor',
    # The official SUMMIT utility defaults to 0.75 for ViT layer decay.
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
