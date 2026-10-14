import os
import subprocess
import sys
import time
from pathlib import Path
from mmengine.config import Config
from collections.abc import Mapping

# runnings.py 位于仓库根目录，因此不用硬编码仓库路径
repo_root = Path(__file__).resolve().parent

experiments = [

#      {"name": "pie_rgb_ownstats_itpn_okd_ld085",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.85}},

#     {"name": "pie_rgb_ownstats_itpn_okd_ccl_ld085",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.85}},
    
#      {"name": "pie_sar_ownstats_itpn_okd_ld085",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.85}},

#     {"name": "pie_sar_ownstats_itpn_okd_ccl_ld085",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.85}},
    
    
    
# 使用另一份 config：
    # {
    #     "name": "pie_rgb_other_config_frozen",
    #     "config": "configs/_major_/another_config.py",
    #     "checkpoint": "G:/weights/another_checkpoint.pth",
    #     "frozen": False,
    #     "cfg_options": {
    #         # 还可以灵活覆盖其他参数
    #          # "optim_wrapper.optimizer.lr": 3e-5,
    #          # "train_cfg.max_iters": 8000,
    #          # "
    # },


    
    ##########################################################################################
#     {"name": "pie_rgb_ownstats_decshare_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "pie_rgb_ownstats_okd_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_rgb_ownstats_okd_ccl_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_rgb_ownstats_okd_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixel_pixelC/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_rgb_ownstats_okd_ccl_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
        
    # {"name": "pie_sar_ownstats_decshare_ld090_lr3e5_0",
    #  "config": "configs/_major_/pie_sar_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "pie_sar_ownstats_okd_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_sar_ownstats_okd_ccl_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_sar_ownstats_okd_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixel_pixelC/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
#     {"name": "pie_sar_ownstats_okd_ccl_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/pie_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00002}},
    
        ##########################################################################################

#     {"name": "fusarmap_sar_ownstats_onlysar_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8_SAR/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "fusarmap_sar_ownstats_decshare_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "fusarmap_sar_ownstats_okd_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "fusarmap_sar_ownstats_okd_ccl_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "fusarmap_sar_ownstats_okd_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixel_pixelC/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "fusarmap_sar_ownstats_okd_ccl_cdr_ld090_lr3e5_0",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
    
    
    # ablation
    ##########################################################################################
    
    # {"name": "whu_bs8_sar_ownstats_onlysar_ld093_lr3e5_42",
    #  "config": "configs/_major_/whu20_sar_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8_SAR/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    # {"name": "whu_bs8_sar_ownstats_decshare_ld093_lr3e5_42",
    #  "config": "configs/_major_/whu20_sar_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_sar_ownstats_okd_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_sar_ownstats_okd_ccl_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
  
#     {"name": "whu_bs8_sar_ownstats_okd_cdr_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixel_pixelC/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_sar_ownstats_okd_ccl_cdr_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
    
    # {"name": "whu_bs8_rgb_ownstats_onlysar_ld093_lr3e5_0",
    #  "config": "configs/_major_/whu20_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8_RGB/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    # {"name": "whu_bs8_rgb_ownstats_decshare_ld093_lr3e5_0",
    #  "config": "configs/_major_/whu20_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_baseline/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_rgb_ownstats_okd_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_rgb_ownstats_okd_ccl_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_cattncon_CA_woffnproj/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_rgb_ownstats_okd_cdr_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixel_pixelC/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_bs8_rgb_ownstats_okd_ccl_cdr_ld093_lr3e5_42",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
         # Reproduction
    # # ##################################################################
    ## ddhr
    # {"name": "ddhr_rgb_ownstats_gfm_90",
    #  "config": "configs/_reproduce_/ddhr_rgb_ownstats_gfm_swin_b.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/GFM/gfm.pth",
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},

    # {"name": "ddhr_rgb_ownstats_rsva",
    #  "config": "configs/_reproduce_/ddhr_rgb_ownstats_rvsa_vitae_b.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/RVSA/vitae-b-checkpoint-1599-transform-no-average.pth",
    #  "cfg_options": {}},

    # {"name": "ddhr_rgb_ownstats_satlas_90",
    #  "config": "configs/_reproduce_/ddhr_rgb_ownstats_satlas_swin_v2_b.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/SatLas/aerial_swinb_si.pth",
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},

    # {"name": "ddhr_rgb_ownstats_selectivemae",
    #  "config": "configs/_reproduce_/ddhr_rgb_ownstats_selective_mae_vit_b.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/SelectiveMAE/vit-base-opticalrs4m.pth",
    #  "cfg_options": {}},

    # {"name": "ddhr_rgb_ownstats_mars_90",
    #  "config": "configs/_reproduce_/ddhr_rgb_ownstats_mars.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/MaRS/mars_base_rgb_encoder_only_trans_swinofficial.pth",
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},

    # {"name": "ddhr_sar_ownstats_sarjepa",
    #  "config": "configs/_reproduce_/ddhr_sar_ownstats_sar_jepa_vit_b.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/SAR-JEPA/checkpoint-200.pth",
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "ddhr_sar_ownstats_saratrx",
#      "config": "configs/_reproduce_/ddhr_sar_ownstats_saratr_x_hivit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARATR-X/186K_notest_checkpoint-600.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "ddhr_sar_ownstats_summit",
#      "config": "configs/_reproduce_/ddhr_sar_ownstats_summit_vitdet_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SUMMIT/SUMMIT.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "ddhr_sar_ownstats_sarmae",
#      "config": "configs/_reproduce_/ddhr_sar_ownstats_sarmae_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARMAE/SARMAE_vitb_checkpoint-last",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

    # {"name": "ddhr_sar_ownstats_mars",
    #  "config": "configs/_reproduce_/ddhr_sar_ownstats_mars.py",
    #  "checkpoint": "/root/autodl-tmp/reproduce_weights/MaRS/mars_base_sar_encoder_only_trans_swinofficial.pth",
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
    #                  "randomness.seed": 42,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    ## whu
#     {"name": "whu_rgb_mmatrx_ld093_lr3e5_42",
#      "config": "configs/_major_/whu_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu_sar_mmatrx_ld093_lr3e5_42",
#      "config": "configs/_major_/whu_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu20_rgb_ownstats_gfm_90",
#      "config": "configs/_reproduce_/whu_rgb_ownstats_gfm_swin_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/GFM/gfm.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_rgb_ownstats_rsva",
#      "config": "configs/_reproduce_/whu_rgb_ownstats_rvsa_vitae_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/RVSA/vitae-b-checkpoint-1599-transform-no-average.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_rgb_ownstats_satlas_90",
#      "config": "configs/_reproduce_/whu_rgb_ownstats_satlas_swin_v2_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SatLas/aerial_swinb_si.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "whu20_rgb_ownstats_selectivemae",
#      "config": "configs/_reproduce_/whu_rgb_ownstats_selective_mae_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SelectiveMAE/vit-base-opticalrs4m.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_rgb_ownstats_mars",
#      "config": "configs/_reproduce_/whu_rgb_ownstats_mars.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/MaRS/mars_base_rgb_encoder_only_trans_swinofficial.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_sar_ownstats_sarjepa",
#      "config": "configs/_reproduce_/whu_sar_ownstats_sar_jepa_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SAR-JEPA/checkpoint-200.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_sar_ownstats_saratrx",
#      "config": "configs/_reproduce_/whu_sar_ownstats_saratr_x_hivit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARATR-X/186K_notest_checkpoint-600.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_sar_ownstats_summit",
#      "config": "configs/_reproduce_/whu_sar_ownstats_summit_vitdet_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SUMMIT/SUMMIT.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_sar_ownstats_sarmae",
#      "config": "configs/_reproduce_/whu_sar_ownstats_sarmae_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARMAE/SARMAE_vitb_checkpoint-last",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "whu20_sar_ownstats_mars",
#      "config": "configs/_reproduce_/whu_sar_ownstats_mars.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/MaRS/mars_base_sar_encoder_only_trans_swinofficial.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
        
    # mars -- DINOv3 Sat
    # {"name": "controll_osinp_marsdino_ddhr_rgb_ld093_lr3e5_0",
    #  "config": "configs/_major_/ddhr_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_dino/checkpoint-99_rgb.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_marsdino_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "controll_osinp_marsdino_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_dino/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    # fusarmap
    
#     {"name": "fusarmap_sar_mmatrx_ld093_lr3e5_42",
#      "config": "configs/_major_/fusarmap_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_dino_pixC_CA_800E/checkpoint-800.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "fusarmap_sar_ownstats_sarjepa",
#      "config": "configs/_reproduce_/fusarmap_sar_ownstats_sar_jepa_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SAR-JEPA/checkpoint-200.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "fusarmap_sar_ownstats_saratrx",
#      "config": "configs/_reproduce_/fusarmap_sar_ownstats_saratr_x_hivit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARATR-X/186K_notest_checkpoint-600.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "fusarmap_sar_ownstats_summit",
#      "config": "configs/_reproduce_/fusarmap_sar_ownstats_summit_vitdet_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SUMMIT/SUMMIT.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "fusarmap_sar_ownstats_sarmae",
#      "config": "configs/_reproduce_/fusarmap_sar_ownstats_sarmae_vit_b.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/SARMAE/SARMAE_vitb_checkpoint-last",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.75,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},

#     {"name": "fusarmap_sar_ownstats_mars",
#      "config": "configs/_reproduce_/fusarmap_sar_ownstats_mars.py",
#      "checkpoint": "/root/autodl-tmp/reproduce_weights/MaRS/mars_base_sar_encoder_only_trans_swinofficial.pth",
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.90,
#                      "randomness.seed": 42,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    # Controlled comparison
    # # ##################################################################
        
    ## ospretrain-1m inp ddhr
    # single_modality
#     {"name": "controll_osinp_mae_single_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_RGB_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_mae_single_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_SAR_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_mae_single_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_RGB_100e_inp/checkpoint-99.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_SAR_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # dual_modality
#     {"name": "controll_osinp_mae_dual_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_mae_dual_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_mae_dual_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_rgb_ld093_lr3e5_0",
     "config": "configs/_major_/ddhr_rgb_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 0,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_sar_ld093_lr3e5_0",
     "config": "configs/_major_/ddhr_sar_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 0,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_rgb_ld093_lr3e5_27",
     "config": "configs/_major_/ddhr_rgb_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 27,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_sar_ld093_lr3e5_27",
     "config": "configs/_major_/ddhr_sar_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 27,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_rgb_ld093_lr3e5_42",
     "config": "configs/_major_/ddhr_rgb_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 42,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    {"name": "ospretrain1mall_canny_ddhr_sar_ld093_lr3e5_42",
     "config": "configs/_major_/ddhr_sar_ownstats.py",
     "checkpoint": "/root/autodl-tmp/weights/ospretrain_all_canny_100e/checkpoint-99.pth",
     "frozen": False,
     "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
                     "randomness.seed": 42,
                     "optim_wrapper.optimizer.lr": 0.00003}},
    
    # okd
#     {"name": "controll_osinp_okd_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_okd_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_okd_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
            # codemae
#     {"name": "controll_osinp_codemae_ddhr_rgb_ld095_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_codemae_ddhr_sar_ld095_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_codemae_ddhr_rgbsar_ld095_lr3e5_0",
    #  "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    # codemae + mlp
#     {"name": "controll_osinp_codemaemlp_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_codemaemlp_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_codemaemlp_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
    # codemae + mlp whu20
#     {"name": "controll_osinp_codemaemlp_whu20_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_codemaemlp_whu20_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_codemaemlp_whu20_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/ospretrain_paired_pixC_cclmlp_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    # mars
#     {"name": "controll_osinp_marsTrue_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_marsTrue_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_marsTrue_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#         # mars + dino
#     {"name": "controll_osinp_marsdinoTrue_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_marsdinoTrue_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_marsdinoTrue_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    

        
    # msgfm
#     {"name": "controll_osinp_msgfm_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_msgfm_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_msgfm_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
        # msgfm + dino
#     {"name": "controll_osinp_msgfmdino_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_msgfmdino_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_msgfmdino_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
            
    # decur
#     {"name": "controll_osinp_decur_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_decur_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_decur_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    # satvit
#     {"name": "controll_osinp_satvit_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_satvit_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_satvit_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
        # satvit + dino
#     {"name": "controll_osinp_satvitdino_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_satvitdino_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_satvitdino_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
        
    # swinssl
#     {"name": "controll_osinp_swinssl_ddhr_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_swinssl_ddhr_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/ddhr_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_swinssl_ddhr_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/ddhr_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
    
    
    ## ospretrain-1m inp whu
    # random
#     {"name": "controll_osinp_random_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_random/checkpoint-0.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_random_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_random/checkpoint-0.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_random_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/itpn_random/checkpoint-0.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    # single_modality
#     {"name": "controll_osinp_mae_single_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_RGB_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_mae_single_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_SAR_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_mae_single_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "rgb_checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_RGB_100e_inp/checkpoint-99.pth",
    #  "sar_checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_SAR_100e_inp/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
        
    # dual_modality
    # {"name": "controll_osinp_mae_dual_whu_rgb_ld093_lr3e5_0",
    #  "config": "configs/_major_/whu20_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_mae_dual_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_mae_dual_whu_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights/INP_dec8share_Paired_100E/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     # okd
#     {"name": "controll_osinp_okd_whu20_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_okd_whu20_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_okd_whu20_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_okd_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    # codemae
#     {"name": "controll_osinp_codemae_whu20_rgb_ld095_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_codemae_whu20_sar_ld095_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_codemae_whu20_rgbsar_ld095_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
    
    # mars
#     {"name": "controll_osinp_marsTrue_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_marsTrue_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_marsTrue_whu_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
        # mars + dino
#     {"name": "controll_osinp_marsdinoTrue_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_marsdinoTrue_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_marsdinoTrue_whu_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_os_100e_inp_true_dino/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # msgfm
#     {"name": "controll_osinp_msgfm_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_msgfm_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_msgfm_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
            
    # msgfm + dino
#     {"name": "controll_osinp_msgfmdino_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_msgfmdino_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_msgfmdino_whu_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
            
    # decur
#     {"name": "controll_osinp_decur_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_decur_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_decur_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "rgb_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_rgb.pth",
    #  "sar_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp/checkpoint-99_sar.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
    # satvit
#     {"name": "controll_osinp_satvit_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_satvit_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_satvit_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "rgb_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
    #  "sar_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
    
        # satvit + dino
#     {"name": "controll_osinp_satvitdino_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_satvitdino_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
#     {"name": "controll_osinp_satvitdino_whu_rgbsar_ld093_lr3e5_0",
#      "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_dino/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
        
    # swinssl
#     {"name": "controll_osinp_swinssl_whu_rgb_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
    
#     {"name": "controll_osinp_swinssl_whu_sar_ld093_lr3e5_0",
#      "config": "configs/_major_/whu20_sar_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00003}},
        
    # {"name": "controll_osinp_swinssl_whu_rgbsar_ld093_lr3e5_0",
    #  "config": "configs/_controlled_/whu20_rgb_sar_ownstats_independent.py",
    #  "rgb_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_rgb.pth",
    #  "sar_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp/checkpoint-99_sar.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00003}},
   
    
    
    
      ## GeoNRW inp geonrw 100% labels
    # single_modality
    # {"name": "controll_odeminp_mae_single_geonrw_rgb_ld093_lr6e5_0_bs16_010",
    #  "config": "configs/_major_/geonrw100_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/itpn_RGB_mae_100e_inp_geonrw/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00006}},
            
    # {"name": "controll_odeminp_mae_single_geonrm100_rgbdem_ld093_lr6e5_0_bs16_010",
    #  "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
    #  "rgb_checkpoint": "/root/autodl-tmp/weights_controll/itpn_RGB_mae_100e_inp_geonrw/checkpoint-99.pth",
    #  "sar_checkpoint": "/root/autodl-tmp/weights_controll/itpn_DEM_mae_100e_inp_geonrw/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00006}},
    
        
    # codemae
    # {"name": "controll_odeminp_codemae_geonrw_rgb_ld093_lr6e5_0_bs16_010",
    #  "config": "configs/_major_/geonrw100_rgb_ownstats.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp_geonrw/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00006}},
        
    # {"name": "controll_odeminp_codemae_geonrm_rgbdem_ld093_lr6e5_0_bs16_100",
    #  "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp_geonrw/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00006}},
    
    
#     {"name": "controll_odeminp_codemae_geonrm_rgbdem_ld093_lr6e5_0_bs16_100",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.95,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
    # {"name": "controll_odeminp_codemae_geonrm_rgbdem_ld093_lr6e5_0_bs16_100",
    #  "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
    #  "checkpoint": "/root/autodl-tmp/weights_controll/codemae_100e_inp_geonrw/checkpoint-99.pth",
    #  "frozen": False,
    #  "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.97,
    #                  "randomness.seed": 0,
    #                  "optim_wrapper.optimizer.lr": 0.00006}},
    
#         # random
#     {"name": "controll_odeminp_random_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_random/checkpoint-0.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     {"name": "controll_odeminp_random_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_random/checkpoint-0.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
#     # dual_modality
#     {"name": "controll_odeminp_mae_dual_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     {"name": "controll_odeminp_mae_dual_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/itpn_mae_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
# #     okd
# #     {"name": "controll_odeminp_okd_geonrw_rgb_ld093_lr6e5_0_bs16",
# #      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
# #      "checkpoint": "/root/autodl-tmp/weights_controll/okd_100e_inp_geonrw/checkpoint-99.pth",
# #      "frozen": False,
# #      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
# #                      "randomness.seed": 0,
# #                      "optim_wrapper.optimizer.lr": 0.00006}},
        
# #     {"name": "controll_odeminp_okd_geonrm100_rgbdem_ld093_lr6e5_0_bs16",
# #      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
# #      "checkpoint": "/root/autodl-tmp/weights_controll/okd_100e_inp_geonrw/checkpoint-99.pth",
# #      "frozen": False,
# #      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
# #                      "randomness.seed": 0,
# #                      "optim_wrapper.optimizer.lr": 0.00006}},

    
#     # mars
# #     {"name": "controll_odeminp_mars_geonrw_rgb_ld093_lr6e5_0_bs16",
# #      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
# #      "checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_geonrw_wometadistill/checkpoint-99_rgb.pth",
# #      "frozen": False,
# #      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
# #                      "randomness.seed": 0,
# #                      "optim_wrapper.optimizer.lr": 0.00006}},
        
# #     {"name": "controll_odeminp_mars_geonrm100_rgbdem_ld093_lr6e5_0_bs16",
# #      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
# #      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_geonrw_wometadistill/checkpoint-99_rgb.pth",
# #      "sar_checkpoint": "/root/autodl-tmp/weights_controll/mars_100e_inp_geonrw_wometadistill/checkpoint-99_sar.pth",
# #      "frozen": False,
# #      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
# #                      "randomness.seed": 0,
# #                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     # msgfm
#     {"name": "controll_odeminp_msgfm_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     {"name": "controll_odeminp_msgfm_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/msgfm_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
            
#     # decur
#     {"name": "controll_odeminp_decur_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp_geonrw/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
        
#     {"name": "controll_odeminp_decur_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp_geonrw/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/decur_100e_inp_geonrw/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
#     # satvit
#     {"name": "controll_odeminp_satvit_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     {"name": "controll_odeminp_satvit_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_geonrw/checkpoint-99.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/satvit_100e_inp_geonrw/checkpoint-99.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    
        
#     # swinssl
#     {"name": "controll_odeminp_swinssl_geonrw_rgb_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_major_/geonrw100_rgb_ownstats.py",
#      "checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp_geonrw/checkpoint-99_rgb.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
        
#     {"name": "controll_odeminp_swinssl_geonrm_rgbdem_ld093_lr6e5_0_bs16_010",
#      "config": "configs/_controlled_/geonrw100_rgb_dem_ownstats_independent.py",
#      "rgb_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp_geonrw/checkpoint-99_rgb.pth",
#      "sar_checkpoint": "/root/autodl-tmp/weights_controll/swinssl_100e_inp_geonrw/checkpoint-99_sar.pth",
#      "frozen": False,
#      "cfg_options": {"optim_wrapper.paramwise_cfg.decay_rate": 0.93,
#                      "randomness.seed": 0,
#                      "optim_wrapper.optimizer.lr": 0.00006}},
    

    
]


def format_cfg_value(value):
    """转换为 MMEngine --cfg-options 能识别的格式。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def find_backbone_option_paths(backbone_cfg, option_name, prefix="model.backbone"):
    if not isinstance(backbone_cfg, Mapping):
        return []
    paths = []
    if option_name in backbone_cfg:
        paths.append(f"{prefix}.{option_name}")

    for key, value in backbone_cfg.items():
        if isinstance(value, Mapping):
            paths.extend(
                find_backbone_option_paths(
                    value, option_name, f"{prefix}.{key}"))
    return paths

env = os.environ.copy()
num_gpus = 8

env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 确保 dist_train.sh 中的 python 仍然是启动 runnings.py 的 Conda 环境
env["PATH"] = (
    str(Path(sys.executable).parent)
    + os.pathsep
    + env.get("PATH", "")
)

print(f"当前 Python 解释器: {sys.executable}")
print(f"仓库目录: {repo_root}")

for experiment in experiments:
    name = experiment["name"]

    config = Path(experiment["config"])
    if not config.is_absolute():
        config = repo_root / config
    
    requested_frozen = experiment.get("frozen")
    work_dir = repo_root / "work_dirs" / name

    if not config.is_file():
        raise FileNotFoundError(f"Config 不存在: {config}")
    
    parsed_config = Config.fromfile(str(config))
    backbone_cfg = parsed_config.get("model", {}).get("backbone", {})
    cfg_options = dict(experiment.get("cfg_options", {}))
   
    cfg_options.setdefault("randomness.seed", 42)
    cfg_options.setdefault("randomness.diff_rank_seed", False)
    cfg_options.setdefault("randomness.deterministic", False)
    cfg_options.setdefault("env_cfg.cudnn_benchmark", False)
    
    # 自动定位单 backbone 或 independent 双 backbone 的 checkpoint。
    # 对双 backbone，可以分别提供 rgb_checkpoint 和 sar_checkpoint；
    # 只提供 checkpoint 时则由两个模态共用。
    shared_checkpoint = experiment.get("checkpoint")
    rgb_checkpoint = experiment.get("rgb_checkpoint", shared_checkpoint)
    sar_checkpoint = experiment.get("sar_checkpoint", shared_checkpoint)
    checkpoint_paths = find_backbone_option_paths(
        backbone_cfg, "checkpoint")
    if not checkpoint_paths:
        # 兼容原来没有在 config 中预先声明 checkpoint 的单 backbone。
        checkpoint_paths = ["model.backbone.init_cfg.checkpoint"]

    applied_checkpoint_overrides = {}
    for checkpoint_path in checkpoint_paths:
        if ".rgb_backbone." in checkpoint_path:
            checkpoint_value = rgb_checkpoint
        elif ".sar_backbone." in checkpoint_path:
            checkpoint_value = sar_checkpoint
        else:
            checkpoint_value = shared_checkpoint

        if checkpoint_value is None:
            raise KeyError(
                f"实验 {name} 缺少 {checkpoint_path} 对应的 checkpoint。"
                "单 backbone 请设置 checkpoint；independent 双 backbone "
                "请设置 rgb_checkpoint 和 sar_checkpoint，或设置一个共用的 "
                "checkpoint。")

        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint 不存在 ({checkpoint_path}): {checkpoint}")
        checkpoint_value = checkpoint.as_posix()
        cfg_options[checkpoint_path] = checkpoint_value
        applied_checkpoint_overrides[checkpoint_path] = checkpoint_value

    # 顶层变量不控制初始化，只用于让自动保存的 config 清楚记录来源。
    if any(".rgb_backbone." in path for path in checkpoint_paths):
        cfg_options["rgb_pretrained"] = Path(rgb_checkpoint).as_posix()
    if any(".sar_backbone." in path for path in checkpoint_paths):
        cfg_options["sar_pretrained"] = Path(sar_checkpoint).as_posix()
    if not any(
            marker in path for path in checkpoint_paths
            for marker in (".rgb_backbone.", ".sar_backbone.")):
        cfg_options["pretrained"] = Path(shared_checkpoint).as_posix()

    # ``frozen`` 不是 MMSeg backbone 的统一接口。只覆盖 config 中已经
    # 声明该参数的 backbone，避免给 GFM、Satlas、SelectiveMAE 等模型
    # 传入其构造函数不接受的参数。
    frozen_paths = find_backbone_option_paths(backbone_cfg, "frozen")
    use_checkpoint_paths = set(
        find_backbone_option_paths(backbone_cfg, "use_checkpoint"))
    applied_frozen_paths = []
    applied_checkpoint_paths = []
    if requested_frozen is not None and frozen_paths:
        for frozen_path in frozen_paths:
            cfg_options[frozen_path] = requested_frozen
            applied_frozen_paths.append(frozen_path)

            # 仅联动同一个 backbone 下已经存在的 use_checkpoint。
            checkpoint_path = (
                f"{frozen_path.rsplit('.', 1)[0]}.use_checkpoint")
            if checkpoint_path in use_checkpoint_paths:
                cfg_options[checkpoint_path] = not requested_frozen
                applied_checkpoint_paths.append(checkpoint_path)

    if requested_frozen is not None and not frozen_paths:
        print(
            f"提示: {name} 的 backbone 没有 frozen 接口，"
            f"已忽略 frozen={requested_frozen}，按 config 原设置训练。")
    
    
#     # 真正控制初始化 checkpoint
#     cfg_options["model.backbone.init_cfg.checkpoint"] = (
#         checkpoint.as_posix()
#     )

#     # 顶层 pretrained 主要用于保存的配置内容保持一致
#     cfg_options["pretrained"] = checkpoint.as_posix()

#     cfg_options["model.backbone.frozen"] = frozen
#     cfg_options["model.backbone.use_checkpoint"] = not frozen

    command = [
    "bash",
    str(repo_root / "tools" / "dist_train.sh"),
    str(config),
    str(num_gpus),
    "--work-dir",
    str(work_dir),
    "--cfg-options",
    ]

    command.extend(
        f"{key}={format_cfg_value(value)}"
        for key, value in cfg_options.items()
    )

    print("=" * 80)
    print(f"实验名称: {name}")
    print(f"Config: {config}")
    for checkpoint_path, checkpoint_value in (
            applied_checkpoint_overrides.items()):
        print(f"Checkpoint: {checkpoint_path}={checkpoint_value}")
    if applied_frozen_paths:
        print(
            f"Frozen: {requested_frozen} -> "
            f"{', '.join(applied_frozen_paths)}")
    else:
        print("Frozen: 未覆盖（backbone 按 config 原设置）")
    if applied_checkpoint_paths:
        print(
            f"use_checkpoint: {not requested_frozen} -> "
            f"{', '.join(applied_checkpoint_paths)}")
    print(f"Work dir: {work_dir}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        shell=False,
    )

    if result.returncode != 0:
        print(f"⚠️ 实验执行失败: {name}")
        print(f"返回码: {result.returncode}")
        break

    print(
        f"✅ 实验完成: {name}\n"
        f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    
# subprocess.run("/usr/bin/shutdown", shell=True)