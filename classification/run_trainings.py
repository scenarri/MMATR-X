import subprocess
import os
import time

commands = [
'python main_finetune.py --model itpn_base_dual --layer_decay 0.85 --drop_path 0.1 --blr 1e-3 --epochs 100 --warmup_epochs 5 --seed 0 --batch_size 32  --dataset AID --data_path F:/TESTDATASETS/AID --data_split 20 --nb_classes 30 --modality RGB',
'python main_finetune.py --model itpn_base_dual --layer_decay 0.95 --drop_path 0.1 --blr 1e-3 --epochs 100 --warmup_epochs 5 --seed 0 --batch_size 32  --dataset NWPURESISC45 --data_path F:/TESTDATASETS/NWPU-RESISC45 --data_split 10 --nb_classes 45 --modality RGB',


'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset FUSAR --data_path F:/TESTDATASETS/SARCls/New_FUSAR --data_split 40 --nb_classes 10 --modality SAR',

'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 0.85 --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset MSTAR --data_path F:/TESTDATASETS/SARCls/MSTAR_SOC --data_split 40 --nb_classes 10 --modality SAR',

'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 10 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 0 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 1 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 2 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 3 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 40 --nb_classes 10 --modality SAR',
'python main_finetune.py --smoothing 0. --mixup 0. --cutmix 0. --layer_decay 1. --drop_path 0.05 --blr 1e-3 --epochs 60 --warmup_epochs 2 --seed 4 --batch_size 50  --dataset SARACD --data_path F:\\TESTDATASETS\\SARCls\\SAR_ACD --data_split 40 --nb_classes 10 --modality SAR',


]

for cmd in commands:
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Time: {t1}, running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️ failure: {cmd}")
        break
    else:
        t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Time: {t2}, ✅ done: {cmd}\n")
