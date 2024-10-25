
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
from data.data_finetune import build_loader_metalens
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from apex import amp


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

    parser.add_argument('--eval_output', default='evaluation_output', type=str,
                        help='path to place evaluation results (not model files but data files)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')

    # newly added for metalens
    parser.add_argument('--target_file', help='you can designate a specific txt file as the initial JM array, or'
                                              'this code will automatically find the latest one')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    # evaluation/metasurface_design/JM_generator to load targeted npy file
    recon_path = config.RECON_PATH + '/reconJMs/'
    if config.LENS.TARGET == '':
        config.defrost()
        config.LENS.TARGET = find_latest_file(recon_path)
        logger.info(f"Have found the latest metalens JM file: {config.LENS.TARGET}")
        config.freeze()
    origin_file = recon_path + config.LENS.TARGET
    if os.path.exists(origin_file):
        source_JM = np.loadtxt(origin_file)
    else:
        raise ValueError("No satisfactory txt file!")

    # directly use sqrt to unpack size
    size = int(np.sqrt(source_JM.shape[0]))
    source_JM = source_JM.reshape((source_JM.shape[0], config.DATA.SIZE_X, config.DATA.SIZE_Y))
    norm_para = prediction(config, source_JM)
    real_para, max_size = realpara_generate(norm_para)
    new_para, new_unitx, new_unity = optimized(real_para, source_JM, max_size, config)
    print('final pitch x:', new_unitx)
    print('final pitch y:', new_unity)
    with open(config.LENS.OUTPUT + '/final_pitch.txt', 'w') as file:
        file.write(f"{new_unitx * 1e9}\n")
        file.write(f"{new_unity * 1e9}\n")
    new_para = truncate(new_para, new_unitx, size)
    np.savetxt(config.LENS.OUTPUT + '/final_para.txt', new_para, fmt='%.3f')
    final_aperture(new_unity * 1e9, int(np.sqrt(len(new_para))))


def final_aperture(periodY, count):
    len_radius = periodY * count / 2
    print(len_radius)

    first_part = [[1.5 * len_radius, 0], [1.5 * len_radius, -1.5 * len_radius], [-1.5 * len_radius, -1.5 * len_radius],
                  [-1.5 * len_radius, 1.5 * len_radius], [1.5 * len_radius, 1.5 * len_radius], [1.5 * len_radius, 0]]

    angles = np.linspace(0, 360, num=200, endpoint=True)
    second_part = np.zeros((len(angles), 2))
    for i in range(len(angles)):
        second_part[i, 0] = len_radius * np.cos(np.radians(angles[i]))
        second_part[i, 1] = len_radius * np.sin(np.radians(angles[i]))

    whole = np.concatenate((first_part, second_part), axis=0)
    np.savetxt(config.LENS.OUTPUT + '/aperture.txt', whole * 1e-9)


def generateJM(JM_in_Lens, config):
    wavelength_path = config.DATA.PATH + config.DATA.FOLDER_NAME + '/real_wavelengths.txt'
    wavelengths = np.loadtxt(wavelength_path)
    wavelengths = np.array(wavelengths) * 1e-9
    num_waves = len(wavelengths)
    size = int(np.sqrt(JM_in_Lens.shape[0]))
    JM_in_Lens = JM_in_Lens.reshape((size, size, config.DATA.SIZE_X, config.DATA.SIZE_Y))
    amp_bias_adjustment(JM_in_Lens[:, :, :, 0:3], config, num_waves)
    for index, wave in enumerate(wavelengths):
        # you can designate your focus length in config.py. It would be better to be consistent with the focus setting
        # in evaluation/metasurface_design/JM_generator.py
        JM_in_Lens[:, :, index, 3] = LensPhase(size, size, config.LENS.X_UNIT, config.LENS.Y_UNIT, config.LENS.FOCUS, 0, 0, wave) / np.pi
        JM_in_Lens[:, :, index, 4] = LensPhase(size, size, config.LENS.X_UNIT, config.LENS.Y_UNIT, config.LENS.FOCUS, 0, 0, wave) / np.pi
        JM_in_Lens[:, :, index, 5] = LensPhase(size, size, config.LENS.X_UNIT, config.LENS.Y_UNIT, config.LENS.FOCUS, 0, 0, wave) / np.pi

    JM_flatten = JM_in_Lens.reshape((-1, JM_in_Lens.shape[2], JM_in_Lens.shape[3]))
    return JM_flatten


def amp_bias_adjustment(JM, config, num_waves, old_min=-1, old_max=1):
    info_file = config.DATA.PREFOLDER_NAME + '/min_max_mean_list.txt'
    stats = np.loadtxt(info_file).reshape((3, 6, num_waves))
    _min, _max, _mean = stats[0, :, :], stats[1, :, :], stats[2, :, :]
    # config.lens.bias is hard-coded into config.py. This variable is the same as args.bias in evaluation/metasurface_design/main.py
    shrink_min = _mean + (_min - _mean) * (1 - config.LENS.BIAS)
    shrink_max = _mean + (_max - _mean) * (1 - config.LENS.BIAS)
    # only slice amp part [0:3] and transpose to [num_wave, 3]
    shrink_min, shrink_max = shrink_min[0:3].transpose(), shrink_max[0:3].transpose()
    normalized_JM = (JM - old_min) / (old_max - old_min)
    # broadcasting
    scaled_JM = normalized_JM * (shrink_max - shrink_min) + shrink_min
    return scaled_JM


def LensPhase(shapex, shapey, unitx, unity, focus_length, bias_x, bias_y, wavelength):
    x = np.arange(-shapex / 2, shapex / 2) * unitx  # e.g. -256 -- 255
    y = np.arange(-shapey / 2, shapey / 2) * unity
    X, Y = np.meshgrid(x, y, indexing='ij')
    phase_xy = 2 * np.pi / wavelength * (focus_length - np.sqrt(focus_length ** 2 + (X - bias_x) ** 2 + (Y - bias_y) ** 2))
    return np.angle(np.exp(1j * phase_xy))


def prediction(config, source_JM):
    logger.info(f"Creating model: {config.MODEL.NAME} for iterated metalens design")
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

    load_checkpoint(config, model_without_ddp, optimizer, _, logger)

    dataset_pred, data_loader_pred = build_loader_metalens(source_JM, config, logger)
    predict_para, loss = validate(config, data_loader_pred, model)
    logger.info(f"LOSS of the network on the {len(dataset_pred)} JMs: {loss:.1f}%")
    return predict_para


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
    # images for spec ; target for para
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
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
    return para_pred, loss_meter.avg


def realpara_generate(para):
    info_file = config.DATA.PATH + config.DATA.FOLDER_NAME + 'params_from_preprocess.txt'

    def read_feature_string(string):
        parts = string.split('_')
        if len(parts) >= 3:
            try:
                return [int(parts[-3]), int(parts[-2]), int(parts[-1].split('.')[0])]
            except ValueError:
                return None
        else:
            return None

    content = read_to_list(info_file)
    min_size, max_size, step = None, None, None
    target = ''
    for i in range(len(content)):
        if content[i] == "DATA.SUFFIX":
            target = content[i + 1]
            min_size, max_size, step = read_feature_string(target)
    if min_size is None or max_size is None or step is None:
        raise ValueError("Please check your data preprocessing. The params_from_preprocess.txt was not correctly generated.")
    maximums = np.array([max_size, max_size, 80, max_size, max_size, 80])
    # change to [1, 6] for broadcasting
    maximums = maximums.reshape(1, -1)
    realpara = para * maximums

    columns_to_constrain = [0, 1, 3, 4]
    realpara[:, columns_to_constrain] = np.clip(realpara[:, columns_to_constrain], min_size, max_size)

    # pair up finetune path and corresponding pretrained path
    config.defrost()
    config.DATA.PREFOLDER_NAME = find_corresponding_pretrain(config, target)
    config.freeze()

    return realpara, max_size


def find_corresponding_pretrain(config, target):
    # find corresponding pretrain data
    training_folders = [f for f in os.listdir(config.DATA.PATH) if
                        os.path.isdir(os.path.join(config.DATA.PATH, f)) and f.startswith("training_data")]
    for folder in training_folders:
        params_path = os.path.join(config.DATA.PATH, folder, "params_from_preprocess.txt")
        content_2 = read_to_list(params_path)
        for i in range(len(content_2)):
            if content_2[i] == "DATA.SUFFIX":
                if content_2[i + 1] == target:
                    return config.DATA.PATH + folder
                else:
                    break

    return ''


def read_to_list(path):
    with open(path, 'r') as file:
        content = file.read()
        return content.split()


def optimized(para, source_JM, max_size, config):
    x_1 = para[:, 0] * np.cos(np.radians(para[:, 2])) + para[:, 1] * np.sin(np.radians(para[:, 2]))
    y_1 = para[:, 0] * np.sin(np.radians(para[:, 2])) + para[:, 1] * np.cos(np.radians(para[:, 2]))
    x_2 = para[:, 3] * np.cos(np.radians(para[:, 5])) + para[:, 4] * np.sin(np.radians(para[:, 5]))
    y_2 = para[:, 3] * np.sin(np.radians(para[:, 5])) + para[:, 4] * np.cos(np.radians(para[:, 5]))

    xmax = max(x_1 + x_2) + 50
    ymax = max(y_1.max(), y_2.max())
    # initialize x pitch and y pitch, using the same method in metasurface_design/JM_generator.py
    config.defrost()
    config.LENS.X_UNIT = (max_size + 100) * 2e-9
    config.LENS.Y_UNIT = (max_size + 100) * 1e-9
    config.freeze()
    x_gap = config.LENS.X_UNIT - xmax * 10 ** -9
    y_gap = config.LENS.Y_UNIT - ymax * 10 ** -9
    print('x_gap, y_gap', x_gap, y_gap)
    real_para, new_unitx, new_unity = None, None, None

    round = 0
    while x_gap > config.LENS.UPLIMIT and y_gap > config.LENS.UPLIMIT:
        new_unitx = config.LENS.X_UNIT * config.LENS.CHANGE_RATE
        new_unity = config.LENS.Y_UNIT * config.LENS.CHANGE_RATE
        config.defrost()
        config.LENS.X_UNIT = new_unitx
        config.LENS.Y_UNIT = new_unity
        config.freeze()
        JM = generateJM(source_JM, config)
        norm_para = prediction(config, JM)
        real_para, max_size = realpara_generate(norm_para)
        x_1 = para[:, 0] * np.cos(np.radians(para[:, 2])) + para[:, 1] * np.sin(np.radians(para[:, 2]))
        y_1 = para[:, 0] * np.sin(np.radians(para[:, 2])) + para[:, 1] * np.cos(np.radians(para[:, 2]))
        x_2 = para[:, 3] * np.cos(np.radians(para[:, 5])) + para[:, 4] * np.sin(np.radians(para[:, 5]))
        y_2 = para[:, 3] * np.sin(np.radians(para[:, 5])) + para[:, 4] * np.cos(np.radians(para[:, 5]))
        xmax = max(x_1 + x_2) + 50
        ymax = max(y_1.max(), y_2.max())
        x_gap = config.LENS.X_UNIT - xmax * 10 ** -9
        y_gap = config.LENS.Y_UNIT - ymax * 10 ** -9
        round += 1
        print('xmax, ymax, x_gap, y_gap', xmax, ymax, x_gap, y_gap)
        print('iteration times:', round)

    return real_para, new_unitx, new_unity


def truncate(para, unitx, size):
    """
    This method is used to clip array size in case that the FDTD simulation would be very time-consuming
    You could change new_diameter manually. Default value is 30 um
    """
    para = para.reshape((size, size, 6))
    new_diameter = 30e-6
    count = 0
    for i in range(size):
        if i * unitx > new_diameter:
            count = i
            break
    x_1 = - count // 2 + size // 2
    x_2 = count // 2 + size // 2
    y_1 = -count + size
    y_2 = count + size
    print('x_1', x_1, 'x_2', x_2, 'y_1', y_1, 'y_2', y_2)
    mini_para = para[x_1: x_2, y_1: y_2, :]
    print(np.shape(mini_para))
    mini_para_2D = mini_para.reshape(mini_para.shape[0] * mini_para.shape[1], 6)
    print(np.shape(mini_para_2D))
    return mini_para_2D


def find_latest_file(directory='.'):
    today = datetime.date.today()
    latest_date = None
    latest_file = None

    for filename in os.listdir(directory):
        if filename.startswith('type_4_') and filename.endswith('.txt'):
            date_str = filename[7:-4]
            try:
                file_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                if latest_date is None or (today - file_date) < (today - latest_date):
                    latest_date = file_date
                    latest_file = filename
            except ValueError:
                continue

    return latest_file


if __name__ == '__main__':
    _, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(config.LENS.OUTPUT, exist_ok=True)
    rank_indicator = 0
    logger = create_logger(output_dir=config.LENS.OUTPUT, dist_rank=rank_indicator, name=f"{config.LENS.NAME}")
    main(config)
