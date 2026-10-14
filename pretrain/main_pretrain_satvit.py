# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models.models_itpn_mim_satvit as satvit

import math
import sys
from typing import Iterable
import util.lr_sched as lr_sched

import torch.utils.data as data
from opt_sar_dataset import PAIRED_Dataset, Single_Modality_Dataset, MixedBatchSampler


_FLOPS_PROFILE_COMPLETED = False

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='itpn_base_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    parser.add_argument('--modality', default='Both', type=str)

    parser.add_argument('--decoder_depth', default=8, type=int)
    parser.add_argument('--INP', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')

    # Optimizer parameters
    parser.add_argument('--dino', action='store_true')  # enabling distillation from dinov3_sat
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=15, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='ospretrain', type=str)
    parser.add_argument('--data_path', default='G:/SAR_RGB_Pretraining', type=str,
                        help='dataset path')
    parser.add_argument('--data_resume', default=None, type=str, help='load fixed part of the whole dataset')
    parser.add_argument('--exclude_test', default=['WHU-OPT-SAR', 'OGSOD-2', 'DDHR', 'DFC23_Track1'], type=str, help='')
    parser.add_argument('--exclude_dataset', default=['PIE-RGB-SAR', 'YESeg-OPT-SAR', 'SS4EO-S12', 'Unpaired'],
                        type=str, help='')

    parser.add_argument('--output_dir', default='satvit',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def load_INP(args, model):
    if 'base' in args.model:
        ckpt = torch.load('./IN_weights/itpn_base_fpn256.pth', map_location='cpu')
    elif 'large' in args.model:
        ckpt = torch.load('./IN_weights/itpn_large_fpn256.pth', map_location='cpu')
    else:
        raise ValueError()
    new_state_dict = {}
    for key, value in ckpt.items():
        if not key.startswith('mask') and not key.startswith('decoder'):

            if key == 'patch_embed.proj.weight':
                value = torch.cat([value, value], dim=1)
                new_state_dict[key] = value
            elif key == 'patch_embed.proj.bias':
                new_state_dict[key] = value
            elif key == 'patch_embed.norm.weight':
                new_state_dict[key] = value
            elif key == 'patch_embed.norm.bias':
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.log_dir is None:
        args.log_dir = args.output_dir

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if args.dataset == 'ospretrain':
        from opt_sar_dataset import PAIRED_Dataset, Single_Modality_Dataset, MixedBatchSampler
        dataset_train = PAIRED_Dataset(args.data_path, mode='pair', exclude_dataset=args.exclude_dataset,
                                       exclude_test=args.exclude_test, img_size=args.input_size, use_aug=True,
                                       resume=args.data_resume)
    elif args.dataset == 'geonrw':
        from geonrw_dataset import GEONRW_Dataset
        dataset_train = GEONRW_Dataset(args.data_path, img_size=args.input_size, use_aug=True, )

    elif args.dataset == 'sunrgbd':
        from sunrgbd_dataset import SUNRGBD_Dataset
        dataset_train = SUNRGBD_Dataset(args.data_path, img_size=args.input_size, use_aug=True)

    else:
        raise ValueError('dataset must be one of [ospretrain, geonrw, sunrgbd]')

    print(len(dataset_train))

    num_tasks = misc.get_world_size()
    # sampler_train = MixedBatchSampler(dataset_train, batch_size=args.batch_size,
    #                                   num_replicas=num_tasks, shuffle=True, drop_last=True)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = data.DataLoader(
        dataset=dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = satvit.itpn_base_dec512d8b(norm_pix_loss=args.norm_pix_loss,
                                       modality='Both',
                                        dino=args.dino,
                                       decoder_depth=args.decoder_depth,
                                       use_checkpoint=args.use_checkpoint)
    if args.INP:
        load_INP(args, model)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    if args.modality == 'RGB' or args.modality == 'SAR':
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    elif args.modality == 'Both':
        eff_batch_size = (2 * args.batch_size) * args.accum_iter * misc.get_world_size()
    else:
        raise NotImplementedError

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    if args.dino:
        if args.dataset in ['ospretrain', 'geonrw']:
            import create_dinov3_sat
            dino_tea = create_dinov3_sat.dino_distill().to(device)
        elif args.dataset == 'sunrgbd':
            import create_dinov3_lvd
            dino_tea = create_dinov3_lvd.dino_distill().to(device)
        else:
            raise NotImplementedError
    else:
        dino_tea = None

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # save_model(
        #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #     loss_scaler=loss_scaler, epoch=epoch)
        #
        # import models
        # ckpt = torch.load('./output_major/satvit/checkpoint-0.pth', map_location='cpu')
        # model = models.models_itpn_dual.itpn_base_dual(modality='RGB')
        # msg = model.load_state_dict(ckpt['model'], strict=False)
        # print(msg)
        # model = models.models_itpn_dual.itpn_base_dual(modality='SAR')
        # msg = model.load_state_dict(ckpt['model'], strict=False)
        # print(msg)
        # print()

        # data_loader_train.dataset.set_epoch(epoch)
        # data_loader_train.dataset.shuffle()

        train_stats = train_one_epoch(
            model, dino_tea, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and (epoch + 1 == args.epochs) and (
                epoch != 0):
            save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def _sum_profile_flops(profiler):
    return float(sum(event.flops or 0 for event in profiler.key_averages()))

def _report_distributed_flops(teacher_flops, model_flops, local_pairs, device):
    totals = torch.tensor(
        [teacher_flops, model_flops, float(local_pairs)],
        dtype=torch.float64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

    if misc.is_main_process():
        teacher_total, model_total, global_pairs = totals.tolist()
        combined_total = teacher_total + model_total
        world_size = misc.get_world_size()

        print("\n========== FORWARD FLOPs PROFILE (ONE REAL BATCH) ==========")
        print(f"world size: {world_size}")
        print(f"local pairs per rank: {local_pairs}")
        print(f"global pairs in profiled batch: {int(global_pairs)}")
        print(f"DINO teacher cluster total: {teacher_total / 1e12:.6f} TFLOPs")
        print(f"student/model cluster total: {model_total / 1e12:.6f} TFLOPs")
        print(f"combined cluster total: {combined_total / 1e12:.6f} TFLOPs")
        if global_pairs > 0:
            print(f"DINO teacher per pair: {teacher_total / global_pairs / 1e9:.6f} GFLOPs/pair")
            print(f"student/model per pair: {model_total / global_pairs / 1e9:.6f} GFLOPs/pair")
            print(f"combined per pair: {combined_total / global_pairs / 1e9:.6f} GFLOPs/pair")
        print("Scope: forward only; DDP/all-gather communication and backward are not FLOPs-counted.")
        print("Counter note: torch.profiler estimates FLOPs only for supported operators.")
        print("=============================================================\n")

def train_one_epoch(model: torch.nn.Module,
                    teacher: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, ):

    global _FLOPS_PROFILE_COMPLETED

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        should_profile = not _FLOPS_PROFILE_COMPLETED
        teacher_flops = 0.0


        if args.dino:
            with torch.no_grad():
                if should_profile:
                    with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=False,
                            with_stack=False,
                            with_flops=True) as teacher_profiler:
                        with torch.cuda.amp.autocast():
                            tea_inp = samples['original'][:, :3, :, :].cuda(non_blocking=True)
                            tea_out = teacher(tea_inp)
                            tea_feat = tea_out.clone()
                    teacher_flops = _sum_profile_flops(teacher_profiler)
                else:
                    with torch.cuda.amp.autocast():
                        tea_inp = samples['original'][:, :3, :, :].cuda(non_blocking=True)
                        tea_out = teacher(tea_inp)
                        tea_feat = tea_out.clone()
        else:
            tea_feat = None

        if should_profile:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=True) as model_profiler:
                with torch.cuda.amp.autocast():
                    self_loss, distill_loss, _, _ = model(samples, mask_ratio=args.mask_ratio, tea_feat=tea_feat)
            model_flops = _sum_profile_flops(model_profiler)

            # EFFICIENCY PROFILE ADDED: ``normalized`` has one row per paired
            # sample; summing this count across ranks gives the true global batch.
            local_pairs = samples['normalized'].shape[0]
            _report_distributed_flops(
                teacher_flops=teacher_flops,
                model_flops=model_flops,
                local_pairs=local_pairs,
                device=device,
            )
            _FLOPS_PROFILE_COMPLETED = True
        else:
            with torch.cuda.amp.autocast():
                self_loss, distill_loss, _, _ = model(samples, mask_ratio=args.mask_ratio, tea_feat=tea_feat)

        loss = self_loss + distill_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(self_loss=self_loss.item())
        metric_logger.update(distill_loss=distill_loss.item() if distill_loss != 0. else 0.)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        if log_writer is not None and data_iter_step == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:

            ckpt = model_without_ddp.state_dict()
            new_state_dict = {}
            for k, v in ckpt.items():
                if k == 'patch_embed.proj.weight':
                    new_state_dict[k] = v

                    new_k = 'patch_embed_rgb.proj.weight'
                    new_v = v[:, :3, :, :]
                    new_state_dict[new_k] = new_v

                    new_k = 'patch_embed_sar.proj.weight'
                    new_v = v[:, 3:, :, :]
                    new_state_dict[new_k] = new_v

                elif k == 'patch_embed.proj.bias':
                    new_state_dict[k] = v

                    new_state_dict['patch_embed_rgb.proj.bias'] = v
                    new_state_dict['patch_embed_sar.proj.bias'] = v

                elif k == 'patch_embed.norm.weight':
                    new_state_dict[k] = v

                    new_state_dict['patch_embed_rgb.norm.weight'] = v
                    new_state_dict['patch_embed_sar.norm.weight'] = v

                elif k == 'patch_embed.norm.bias':
                    new_state_dict[k] = v

                    new_state_dict['patch_embed_rgb.norm.bias'] = v
                    new_state_dict['patch_embed_sar.norm.bias'] = v

                else:
                    new_state_dict[k] = v

            to_save = {
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if model_ema is not None:
                to_save.update({'model_ema': model_ema.module.state_dict()})

            misc.save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = os.path.join('/root/autodl-tmp/output_major', args.output_dir)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
