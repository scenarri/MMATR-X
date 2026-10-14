import subprocess
import os
import time

commands = [
    
    
    
#     'PORT=25001 bash tools/dist_test.sh ./configs/_itpn_comp_/mars_loveda_ld090_dp010_lr6en5_8xb2_10k.py ../work_dirs/mars_loveda_ld090_dp010_lr6en5_8xb2_10k/iter_10000.pth 8 --cfg-options test_evaluator.format_only=True --out ./mars_loveda_ld090_dp010_lr6en5_8xb2_10k', 
#     'python loveda_label_trans.py --dir mars_loveda_ld090_dp010_lr6en5_8xb2_10k',
#     'zip -r mars_loveda_ld090_dp010_lr6en5_8xb2_10k.zip mars_loveda_ld090_dp010_lr6en5_8xb2_10k/',
#     'rm -r ./mars_loveda_ld090_dp010_lr6en5_8xb2_10k',
    
#     'PORT=25001 bash tools/dist_test.sh ./configs/_itpn_comp_/mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape.py ../work_dirs/mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape/iter_10000.pth 8 --cfg-options test_evaluator.format_only=True --out ./mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape', 
#     'python loveda_label_trans.py --dir mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape',
#     'zip -r mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape.zip mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape/',
#     'rm -r ./mars_loveda_ld090_dp010_lr6en5_8xb2_10k_ape',
    
    'PORT=25001 bash tools/dist_test.sh ./configs/_itpn_comp_/mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape.py ../work_dirs/mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape/iter_80000.pth 8 --cfg-options test_evaluator.format_only=True --out ./mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape', 
    'python loveda_label_trans.py --dir mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape',
    'zip -r mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape.zip mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape/',
    'rm -r ./mars_loveda_ld090_dp010_lr6en5_8xb1_80k_ape',
    
    ]

for cmd in commands:
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Time: {t1}, 正在执行: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️ 命令执行失败: {cmd}")
        break
    else:
        t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Time: {t2}, ✅ 命令执行完成: {cmd}\n")

# subprocess.run("/usr/bin/shutdown", shell=True)