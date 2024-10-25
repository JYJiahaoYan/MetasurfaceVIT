
import os
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, will be written from preprocess
_C.DATA.PATH = './preprocess/'
_C.DATA.FOLDER_NAME = ''
# path for corresponding pretrained dataset.mainly used in main_metalens.py, will be populated automatically through pairing with your finetune dataset
_C.DATA.PREFOLDER_NAME = ''
_C.DATA.PREFIX_JM = ''
_C.DATA.PREFIX_PARAM = ''
_C.DATA.SUFFIX = ''
# Input sizeX need to overwrite using settings from preprocess
_C.DATA.SIZE_X = 20
_C.DATA.SIZE_Y = 6
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# todo Number of data loading threads need to distinguish between pretrain and finetune
_C.DATA.NUM_WORKERS = 4
# mask type having int value from 0 to 5. 0 means randomly using 1-5 types' masks
_C.DATA.MASK_TYPE = 1
# divide num will be populated from preprocess
_C.DATA.DIVIDE_NUM = None
# since training data are too heavy for cache and memory, load different types of data step by step during pretraining.
_C.DATA.DATA_SIZE = 3
_C.DATA.DATA_START = 1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'metaVIT'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of size parameters. For finetuning
_C.MODEL.NUM_PARA = 6
# todo haven't done thorough grid searching for following params:
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
# todo personalized setting, determining using which part of jones matrix to calculate loss.
# 0 for calculating loss of the whole jones matrix
# 1 for calculating loss of masked part
# 2 for calculating loss of unmasked part
_C.MODEL.LOSS_TYPE = 0

# Vision Transformer parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 1
_C.MODEL.VIT.IN_CHANS = 1
_C.MODEL.VIT.EMBED_DIM = 512
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.INIT_VALUES = 0.1
_C.MODEL.VIT.USE_APE = True
_C.MODEL.VIT.USE_RPB = False
_C.MODEL.VIT.USE_SHARED_RPB = False
_C.MODEL.VIT.USE_MEAN_POOLING = False


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.LAYER_DECAY = 1.0

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
# todo may not applicable in this project. This part is borrowed from SIMMIM model.
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
_C.AMP_TYPE = 'apex'

# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
_C.PRETRAINED_OUTPUT = ''

# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 3
# Frequency to logging info
_C.PRINT_FREQ = 50

_C.SEED = 0

_C.EVAL_MODE = False
_C.PREDICT_PARA_PATH = './evaluation/metasurface_verification/predict_params/'
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# todo reconstruct part, may need to open a new branch (_C.RECON.xxx) that follows above style
_C.RECON_MODE = False
_C.RECON_PATH = './evaluation/metasurface_design'
_C.RECON_TYPE = 6
_C.TREATMENT = ''

# metalens part
_C.LENS = CN()
_C.LENS.PREEXIST = True
_C.LENS.NAME = 'metalens'
_C.LENS.OUTPUT = 'metalens_output'
_C.LENS.X_TOTAL = 256
_C.LENS.FOCUS = 75 * 10 ** -6
_C.LENS.X_UNIT = 0.8 * 10 ** -6
_C.LENS.Y_UNIT = 0.4 * 10 ** -6
_C.LENS.CHANGE_RATE = 0.9
_C.LENS.UPLIMIT = 50 * 10 ** -9
_C.LENS.BIAS = 0.3
_C.LENS.TARGET = ''


def update_config(config, args):
    config.defrost()

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('epoch'):
        config.TRAIN.EPOCHS = args.epoch
    if _check_args('mask_type'):
        config.DATA.MASK_TYPE = args.mask_type
    if _check_args('data_size'):
        config.DATA.DATA_SIZE = args.data_size
    if _check_args('data_start'):
        config.DATA.DATA_START = args.data_start
    if _check_args('base_lr'):
        config.TRAIN.BASE_LR = args.base_lr
    if _check_args('warmup_lr'):
        config.TRAIN.WARMUP_LR = args.warmup_lr
    if _check_args('min_lr'):
        config.TRAIN.MIN_LR = args.min_lr
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('data_folder_name'):
        config.DATA.FOLDER_NAME = args.data_folder_name
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_type'):
        config.AMP_TYPE = args.amp_type
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('pretrained_output'):
        config.PRETRAINED_OUTPUT = args.pretrained_output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('recon'):
        config.RECON_MODE = True
    if _check_args('recon_path'):
        config.RECON_PATH = args.recon_path
    if _check_args('recon_type'):
        config.RECON_TYPE = args.recon_type
    if _check_args('treatment'):
        config.TREATMENT = args.treatment
    if _check_args('eval_output'):
        config.PREDICT_PARA_PATH = args.eval_output
    if _check_args('target_file'):
        config.LENS.TARGET = args.target_file

    # update params from preprocess
    if config.DATA.FOLDER_NAME == '':
        config.DATA.FOLDER_NAME = f'training_data_{config.DATA.DATA_START}'
    get_params_from_preprocess(config)
    # update params from command line input (as key-value pairs)
    if args.opts:
        config.merge_from_list(args.opts)

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.PRETRAINED_OUTPUT = os.path.join(config.PRETRAINED_OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_params_from_preprocess(config):
    # basically data-relevant params from preprocess
    fixed_path = config.DATA.PATH + config.DATA.FOLDER_NAME + '/params_from_preprocess.txt'
    if not os.path.exists(fixed_path):
        raise ValueError('Required config file in the folder [preprocess] doesnt exist. You might run preprocess first.')

    with open(fixed_path, 'r') as file:
        content = file.read()
        content = content.split()

    config.merge_from_list(content)


def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config


def get_static_config():
    # don't update config using argparse
    config = _C.clone()
    return config

