import argparse
from image_generator import ImageGenerator
from JM_generator import JMGenerator
import sys
import logging
import os


def get_args_parser():
    parser = argparse.ArgumentParser('MetasurfaceVIT-design', add_help=False)
    # generic params
    parser.add_argument('--pretrain_path', default='./preprocess/training_data_1', type=str,
                        help='The path where your pretrain data exists, with which your designed data should be consistent')
    parser.add_argument('--image_path', default='./figures', type=str,
                        help='The path where you pick some images to generate corresponding Jones Matrix')
    parser.add_argument('--output_path', default='./evaluation/metasurface_design', type=str,
                        help='The path where you save output data files and log files')
    parser.add_argument('--design_type', type=int,
                        help='1 for single-wavelength application; 2 for multi-wavelength app; 3 for RGB three-wave app'
                             '4 for full wavelength metalens application')
    parser.add_argument("--visualize", action='store_true',
                        help="add it if you want to visualize the generated Jones matrices (amplitudes & phases)")
    parser.add_argument('--random_seed', default=1, type=int,
                        help='change it if you want to get JMs from different pictures every time')

    parser.add_argument('--bias', default=0.3, type=float,
                        help='This is an important parameter. For imported images with many 1 and -1 values, they may'
                             'hardly map the real distribution of training data JMs. To loose the extreme constraint on'
                             'amplitudes, use bias to shrink the upper and lower limits of targeted JMs.'
                             'how to do it? say Tmax, Tmin, Tmean mean corresponding values of training data'
                             'for targeted jones matrix, value_after is:'
                             'Tmean +（Tmin - Tmean）* (1-bias)'
                             'Tmean + (Tmax - Tmean)* (1-bias)'
                             'Tmean')
    parser.add_argument('--noise', default=0.0, type=float,
                        help='0 means no noise, every time you run this code, generated jones matrices are the same.'
                             'Add noise means the generated jones matrices could be slightly different.')
    parser.add_argument('--size', default=96, type=int,
                        help="This parameter is x-size (number of pixels along x-axis, mapping to row-axis in my case) "
                             "of a 2d image and corresponding metasurface. Notice the overall dimension would be "
                             "[2*total, total] due to the double structure (size: [2*unit, unit] ) used as each pixel. "
                             "It can be clipped to square for further application.")
    parser.add_argument('--tolerance', default=3, type=int,
                        help="affects the precision of hologram calculation. Larger tolerances mean higher precision "
                             "but longer runtime.")

    # partial params
    parser.add_argument('--fixed_wave', type=parse_fixed_wavelength, default=None,
                        help='Fixed wavelength: none, an integer, or a comma-separated list of integers'
                             'This setting only applies to design type 1 & 2')
    parser.add_argument('--amplitude', default='one', type=str,
                        help='one: means only preserve current wavelength point. apply to design type 1 2 3'
                             'all: means preserve all wavelength channels. apply to type 1, 2, 3, 4'
                             'several: means preserve several wavelength channels. apply to type 2 & 3'
                             'none: do not preserve amplitude, that is, mask all amplitudes and only preserve phase '
                             'part. apply to design type 4')
    parser.add_argument('--handled_wave_per_block', default=1, type=int,
                        help='1 means using 1 block (that is a dimer of silicon nanocubes) to deal with one wavelength channel'
                             '2 means using 1 block to deal with two wavelength channels, and so on.')
    parser.add_argument('--focus_length', default=75, type=int,
                        help="It's applicable for design type 4 (metalens). The default unit is um.")

    return parser


def parse_fixed_wavelength(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return [int(x) for x in value.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid fixed_wave value. Must be 'none', an integer, or a comma-separated list of integers.")


def validate_args(args):
    if args.fixed_wave:
        if (isinstance(args.fixed_wave, int) and args.design_type != 1) or (isinstance(args.fixed_wave, list) and args.design_type != 2):
            raise ValueError(f"The design type {args.design_type} is not matched with the fixed_wave type:{args.fixed_wave}")
    if args.amplitude == 'one' and args.design_type == 4:
        raise ValueError("'one' type amplitude is not applicable to design_type 4.")
    elif args.amplitude == 'several' and args.design_type in (1, 4):
        raise ValueError("'several' type amplitude is not applicable to design_type 1 and 4.")
    elif args.amplitude == 'none' and args.design_type != 4:
        raise ValueError("'None' type only applies to design_type 4.")


def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(datefmt='%Y-%m-%d %H:%M'))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def get_params_from_preprocess(args):
    param_path = args.pretrain_path + '/params_from_preprocess.txt'
    params = {}
    if os.path.exists(param_path):
        with open(param_path, 'r') as file:
            content = file.read()
            content = content.split()
    else:
        raise ValueError("It seems like current folder doesn't contain formal dataset, and no params txt was found.")
    # change list to dict
    for i in range(0, len(content), 2):
        params[content[i]] = content[i + 1]
    return params


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    validate_args(args)
    logger = create_logger(output_dir=args.output_path, name="metasurface_design")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    params = get_params_from_preprocess(args)
    logger.info("Parameters from preprocess and pretrain phases:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")
    # if it's metalens, we only concern focusing issue. no need to load images.
    if args.design_type != 4:
        img_generator = ImageGenerator(args)
        images, norm_images = img_generator.load_images()
    else:
        images, norm_images = None, None

    JM_generator = JMGenerator(args, params, images, norm_images)
    JM_generator.generate_and_save()




