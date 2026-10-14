import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import scipy
import argparse
from tqdm import tqdm
from utils import show, seed_everything, MetricLogger, SmoothedValue
import json
from torch.cuda.amp import autocast as autocast
from get_model import *
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score, MulticlassAccuracy
from get_dataset import *
import lr_sched
from contextlib import nullcontext
import time


def train_one_epoch(model, loader, optimizer, criterion, metrics, epoch, loss_scaler, args):
    model.train()
    for m in metrics:
        metrics[m].reset()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30
    optimizer.zero_grad()
    
    for data_iter, data in enumerate(metric_logger.log_every(loader, print_freq, header)):
        rgb, sar, label = data['rgb'].cuda(), data['sar'].cuda(), data['label'].cuda()
        
        lr_sched.adjust_learning_rate(optimizer, data_iter / len(loader) + epoch, args)
        
        with torch.cuda.amp.autocast() if args.amp else nullcontext():
            logits = model(rgb, sar)
            loss = criterion(logits, label)
        loss_value = loss.item()
        loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False, update_grad=True)
        
        optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        
        for m in metrics:
            metrics[m].update(logits if args.dataset=='EuroSat' else logits.sigmoid(), label)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, loader, metrics, criterion, args):
    model.eval()
    for m in metrics:
        metrics[m].reset()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for data in metric_logger.log_every(loader, 10, header):
        rgb, sar, label = data['rgb'].cuda(), data['sar'].cuda(), data['label'].cuda()
        with torch.cuda.amp.autocast() if args.amp else nullcontext():
            logits = model(data['rgb'].cuda(), data['sar'].cuda())
            loss = criterion(logits, label)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        for m in metrics:
            metrics[m].update(logits if args.dataset=='EuroSat' else logits.sigmoid(), label)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    
    exp_name = '-'.join([args.backbone, args.dataset, args.modality, args.bands, 'amp' if args.amp else '', 'linprob' if args.linprob else 'finetune', '-'+str(args.subset), 'seed', str(args.seed)])
    out_dirs = './output'
    if os.path.exists(os.path.join(out_dirs, f'{exp_name}_log.json')):
        print('Output directory already exists, exiting.')
        return 0
    # os.makedirs(out_dirs, exist_ok=True)
    print(exp_name)
    
    log_data = {
        "exp_config": vars(args),
        "best_metric": 0.0,
        "best_epoch": 0,
        "best_checkpoint_path": "",
        "train_logs": [],
        "eval_logs": [],
    }
    
    g = torch.Generator()
    g.manual_seed(729 + args.seed)
    train_loader, val_loader, num_classes = get_dataloader(args, generator=g)
    args.num_classes = num_classes
    print('train_len: %d val_len: %d' % (len(train_loader.dataset), len(val_loader.dataset)))
    
    model = get_model(args)
    model.cuda()
    
    optimizer = torch.optim.SGD(model.head.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)
    criterion = torch.nn.MultiLabelSoftMarginLoss() if args.dataset != 'EuroSat' else torch.nn.CrossEntropyLoss()
    loss_scaler = lr_sched.NativeScalerWithGradNormCount()
    
    if args.dataset == 'EuroSat':
        metric = {'acc': MulticlassAccuracy(num_classes=num_classes, average='micro').cuda()}
        eval_metric = 'acc'
    else:
        metric = {'microAP': MultilabelAveragePrecision(num_labels=num_classes, average='micro').cuda(),
                  'microF1': MultilabelF1Score(num_labels=num_classes, threshold=0.5, average='micro').cuda()}
        eval_metric = 'microAP'
    best_metric = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, metric, epoch, loss_scaler, args)
        epoch += 1
        for m in metric:
            train_stats[m] = metric[m].compute().item() * 100.
        train_stats['epoch'] = epoch
        log_data["train_logs"].append(train_stats)
        # print(train_stats)
        
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_stats = evaluation(model, val_loader, metric, criterion, args)
            for m in metric:
                eval_stats[m] = metric[m].compute().item() * 100.
            eval_stats['epoch'] = epoch
            log_data["eval_logs"].append(eval_stats)
            # print(eval_stats)
            
            if eval_stats[eval_metric] > best_metric:
                best_metric = eval_stats[eval_metric]
                best_epoch = epoch
                log_data["best_metric"] = best_metric
                log_data["best_epoch"] = best_epoch
                if args.save:
                    torch.save(model.state_dict(), os.path.join(out_dirs, f'best.pth'))
            #print(f'{exp_name}_best_{best_epoch}ep_{best_epoch} {eval_metric}')
    
    print(f'{exp_name}_best_{best_epoch}ep_{best_epoch} {eval_metric}')
    
    log_file = os.path.join(out_dirs, f"{exp_name}_log.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
    
    print(f"Log file saved to: {log_file}, best metric: {best_metric}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='PIE', help='PIE/YESeg/DDHR-SK/WHU/DFC20/BEN/EuroSat')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--valbs', type=int, default=256)
    parser.add_argument('--bands', type=str, default='RGB', choices=['RGB', 'B12', 'B13'])
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'SAR', 'Both'])
    parser.add_argument('--subset', type=float)
    
    # Model & Method
    parser.add_argument('--backbone', type=str, default='SS4EOS12')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    
    # Functioning
    parser.add_argument('--linprob', action='store_true')
    parser.set_defaults(linprob=True)
    
    # Optimizer
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--ld', type=float, default=0.75)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0) # 0, 1, 2, 3, 4
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=True)
    
    args = parser.parse_args()
    
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Time: {t1}")
    
    main(args)