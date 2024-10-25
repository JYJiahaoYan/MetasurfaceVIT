import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter
from config import get_config, get_params_from_preprocess

from model import build_model
# for reconstruction, data_loader is an individual file: data_recon.py
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from apex import amp
from torch.cuda.amp import autocast, GradScaler
# notice: give a choice to choose whether use deprecated apex.amp or torch.cuda.amp
# DeprecatedFeatureWarning: apex.amp is deprecated and will be removed by the end of February 2023.
# Use [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)


def parse_option():
    parser = argparse.ArgumentParser('MetaVIT masked pre-training', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--epoch', type=int, help="total epoch")
    parser.add_argument('--mask_type', type=int,
                        help="mask type for pretraining: 0 for randomly selecting 1-5;"
                             "1 for masking n-1 wavelength channels and only preserving one wavelength's Jones matrix"
                             "2 for keeping all amplitude components (all wavelengths) but only keeping one "
                             "wavelength's phase components"
                             "3 uses the same masking mechanism as 1 but only keeps 11 polarized component and mask"
                             "12 and 22 components"
                             "4 uses the same masking mechanism as 2 but only keeps 11 polarized component and mask"
                             "12 and 22 components"
                             "5 masks 12 and 22 polarized components for all wavelengths but keeps 11 polarized "
                             "components for all wavelengths")
    parser.add_argument('--data_size', type=int,
                        help='1 means the basic. 2 means the amount of basicx2, 3 means amount of basicx3;')
    parser.add_argument('--data_start', type=int,
                        help='if data_start = 1, and data_size=2, it will load training_data_1 and training_data_2'
                             'if data_start = 5, and data_size=1, it will load training_data_5')

    parser.add_argument('--base_lr', type=float, help='learning rate')
    parser.add_argument('--warmup_lr', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume path from checkpoint')
    parser.add_argument('--data_folder_name', help='you could specify the data folder name like training_data_2,'
                                                   'otherwise, it will use the latest')

    parser.add_argument('--accumulation_steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp_type', type=str, default='apex',
                        help="type of automatic mix precision (amp). apex for nvidia apex.amp, pytorch for pytorch.amp")
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--recon", action='store_true', help="whether to run reconstruct mode")
    parser.add_argument('--recon_path', help='path of designed JM and mask for reconstruction')
    parser.add_argument('--recon_type', type=int, help="the design type 1-4 determined during metasurface design")
    parser.add_argument('--treatment', default=None,
                        help="a personalized string to match your designed JM and mask, also will be used to name "
                             "following generated data. Should be a date like '2024-08-20'. Keep None, this code will "
                             "automatically find the data labelled by the latest date.")
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    if config.RECON_MODE:
        data_loader = build_loader(config, logger, "reconstruct")
    else:
        data_loader = build_loader(config, logger, "pre_trained")

    logger.info(f"Creating model: {config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_TYPE == 'apex' and config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    if torch.cuda.device_count() != 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()  # floating-point operations per second
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader))

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

    if config.RECON_MODE:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        loss, JM_recon = validate(config, data_loader, model)
        os.makedirs(config.RECON_PATH + '/reconJMs/', exist_ok=True)
        np.savetxt(config.RECON_PATH + '/reconJMs/type_' + str(config.RECON_TYPE) + '_' + config.TREATMENT + '.txt',
                   JM_recon, fmt='%.3f')
        return

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    daily_time = time.time()
    round_len = config.TRAIN.EPOCHS // config.DATA.DATA_SIZE
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        round_no = epoch // round_len
        if epoch != 0 and epoch % round_len == 0:
            # update data path
            config.defrost()
            config.DATA.FOLDER_NAME = f'training_data_{round_no}/'
            get_params_from_preprocess(config)
            config.freeze()
        if torch.cuda.device_count() != 1:
            data_loader.sampler.set_epoch(epoch)
        train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler)
        if ((torch.cuda.device_count() != 1 and dist.get_rank() == 0) or torch.cuda.device_count() == 1) and (
                epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        daily_period = time.time() - daily_time
        # this part is optional. if you run this program in your office, this part controls the runtime that only runs at night
        if daily_period > 60 * 60 * 12:
            logger.info("Sleeping for nearly 12h...")
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
            time.sleep(60 * 60 * 24 - daily_period)
            daily_time = time.time()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, combine in enumerate(data_loader):
        img, mask = combine
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        if config.AMP_TYPE == 'pytorch':
            with autocast():
                loss = model(img, mask)
        else:
            loss = model(img, mask)
        # accumulation step equals 1 meaning no gradient accumulation in use.
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
        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
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
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    amp = config.DATA.BATCH_SIZE
    JM_recon = np.zeros((len(data_loader) * amp, 1, config.DATA.SIZE_X, config.DATA.SIZE_Y))

    end = time.time()
    # images for spec ; target for para
    for idx, (images, masks) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        loss, recon = model(images, masks)

        loss_meter.update(loss.item(), images.size(0))

        if images.size(0) == amp:
            JM_recon[idx * amp: (idx + 1) * amp, :, :, :] = recon.cpu().numpy()
        else:
            JM_recon[idx * amp: idx * amp + images.size(0), :, :, :] = recon.cpu().numpy()

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
    # reshape as [num_row * num_size * num_size, num_wave * 6]
    JM_recon = JM_recon.reshape((JM_recon.shape[0] * JM_recon.shape[1], JM_recon.shape[2] * JM_recon.shape[3]))
    return loss_meter.avg, JM_recon


if __name__ == '__main__':

    _, config = parse_option()

    # logic to choose between nvidia apex.amp and pytorch.amp
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

    logger.info(config.dump())

    main(config)
