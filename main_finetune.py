
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter

from config import get_config
from model import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from apex import amp
from torch.cuda.amp import autocast, GradScaler


def parse_option():
    parser = argparse.ArgumentParser('MetaVIT finetune&evaluation', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--epoch', type=int, help="total epoch")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='finetune training data path (data_path + data_folder_name)')
    parser.add_argument('--data_folder_name', type=str, help='path to dataset')
    # manually write in resume model path
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp_type', type=str, default='apex',
                        help="type of automatic mix precision (amp). apex for nvidia apex.amp, pytorch for pytorch.amp")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='finetune_output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # the following is pretrained output path:
    parser.add_argument('--pretrained_output', default='output', type=str, metavar='PATH',
                        help='model files. root of pretrained output folder')
    parser.add_argument('--eval_output', type=str,
                        help='path to place evaluation results (not model files but data files)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--recon_type', type=int, help="the design type 1-4 determined during metasurface design")
    parser.add_argument('--treatment', help="manually input the time stamp ( e.g. 2024-10-14) you wanna use for params prediction")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config, logger, type='finetune')
    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    if torch.cuda.device_count() != 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.SmoothL1Loss()

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        if config.EVAL_MODE:
            dataset_pred, data_loader_pred = build_loader(config, logger, type='predict')
            predict_para, loss = validate(config, data_loader_pred, model)
            logger.info(f"LOSS of the network on the {len(dataset_pred)} recon images: {loss:.1f}%")
            # for config.eval_mode, save data and directly return.
            # predict path was hard coded in config.py.
            np.savetxt(config.PREDICT_PARA_PATH + 'type_' + str(config.RECON_TYPE) + '_'
                       + config.TREATMENT + '.txt', predict_para, fmt='%.3f')
            return
        else:
            loss = validate(config, data_loader_val, model)
            logger.info(f"LOSS of the network on the {len(dataset_val)} test images: {loss:.1f}%")

    else:  # starts at the beginning from pretrained mode;
        resume_file = auto_resume_helper(config.PRETRAINED_OUTPUT, logger)
        if resume_file:
            config.defrost()
            config.MODEL.PRETRAIN = resume_file
            config.freeze()
            logger.info(f'start from pretrained: {resume_file}')
        else:
            raise ValueError("can't find relevant files (neither fine-tune resume nor pretrained file)")
        load_pretrained(config, resume_file, model_without_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if torch.cuda.device_count() != 1:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if torch.cuda.device_count() != 1:
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        else:
            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)

        loss = validate(config, data_loader_val, model)
        logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss:.1f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # pytorch amp
        if config.AMP_TYPE == 'pytorch':
            scaler.scale(loss).backward()
            grad_norm = handle_gradient(model.parameters())
        # nvidia apex amp
        elif config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            grad_norm = handle_gradient(amp.master_params(optimizer))
        # no amp
        else:
            loss.backward()
            grad_norm = handle_gradient(model.parameters())

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            if config.AMP_TYPE == "pytorch":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def handle_gradient(parameters):
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.CLIP_GRAD)
    else:
        grad_norm = get_grad_norm(parameters)
    return grad_norm


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    amp = config.DATA.BATCH_SIZE
    if config.EVAL_MODE:
        para_pred = np.zeros((len(data_loader) * amp, 6))
    end = time.time()
    # images for JM ; target for para
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        # print('images size', images.shape)  # 128 1 20 6
        # print('target size', target.shape)  # 128 6
        # print('output size', output.shape)  # 128 6
        # measure accuracy and record loss
        loss = criterion(output, target)
        if config.EVAL_MODE:
            if images.size(0) == amp:
                para_pred[idx * amp: (idx + 1) * amp, :] = output.cpu().numpy()
            else:
                para_pred[idx * amp: idx * amp + images.size(0), :] = output.cpu().numpy()

        if torch.cuda.device_count() != 1:
            loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

    if config.EVAL_MODE:
        return para_pred, loss_meter.avg
    else:
        return loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_TYPE == 'pytorch':
        scaler = GradScaler()
    elif config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if torch.cuda.device_count() != 1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1

        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        seed = config.SEED + dist.get_rank()
    else:
        seed = config.SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    factor = 1 if torch.cuda.device_count() == 1 else dist.get_world_size()
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * factor / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * factor / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    rank_indicator = 0 if torch.cuda.device_count() == 1 else dist.get_rank()
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank_indicator, name=f"{config.MODEL.NAME}")

    if rank_indicator == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
