# definition of metasurface_verification: after we generated params info from fine-tune transformer model, we can verify
# them through actually fabricating metasurfaces or simulating in FDTD. However, they both take times, so here this
# verification part leverages the advantages of neural networks to rapidly achieve forward prediction (from params to
# optical metrics (Jone matrices in this case) and visualize them).
# notice, this metasurface_verification module is not applicable to metalens design.

import argparse
from visualization import Visualizer
from matcher import Matcher
from predictor import Predictor
import os


def get_args_parser():
    parser = argparse.ArgumentParser('MetasurfaceVIT-verification', add_help=False)
    parser.add_argument('--verify_type', default='predictor',
                        help='predictor means it will use neural networks to predict Jones Matrices based on parameters'
                             'matcher means it will traverse existing database to find the closest size-JM pair. May be'
                             'much slower than predictor')
    parser.add_argument('--network', default='MLP',
                        help="if you select predictor as 'verify type', here you should specify whether using MLP or CNN")
    parser.add_argument('--hidden_channels', default=None, nargs="+", type=int,
                        help="keep default, predictor object will automatically fill it. Or you can manually change."
                             "notice, for MLP, this parameter is actually hidden_sizes(widths)")
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate for MLP and CNN")
    parser.add_argument('--epoch', default=100, type=int, help="epoch for MLP and CNN")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size for MLP and CNN")

    parser.add_argument('--design_type', type=int,
                        help='1 for single-wavelength application; 2 for multi-wavelength app; 3 for RGB three-wave app'
                             '4 for full wavelength metalens application')
    parser.add_argument('--treatment', type=str,
                        help='a date label for your data. check evaluation/metasurface_verification/predict_params to'
                             'find the right file and specify its date label. e.g. 2024-10-14')
    parser.add_argument('--path', default='./preprocess/', type=str, help='path for all training data.The specific path should be path + folder')
    parser.add_argument('--finetune_folder', type=str,
                        help='Path of your finetune data. Required if under train mode. You must specify using commandline.')
    parser.add_argument('--pretrain_folder', type=str, help='will be populated by following code.')
    parser.add_argument('--max_size', type=int, help='will be populated by following code.')
    parser.add_argument('--min_size', type=int, help='will be populated by following code.')
    parser.add_argument('--step', type=int, help='will be populated by following code.')
    parser.add_argument('--feature_string', type=str, help='will be populated by following code.')
    parser.add_argument('--angle_step', default=10, type=int,
                        help="if it's not 10, you should manually change it. Be consistent with your training data.")
    parser.add_argument('--params_path', default="./evaluation/metasurface_verification/predict_params")
    parser.add_argument('--designJM_path', default="./evaluation/metasurface_design")

    parser.add_argument('--save_path', default="./evaluation/metasurface_verification/output/",
                        help="the path to place models, loggers, and other output files from this part.")
    parser.add_argument('--train', action='store_true', help="add it if you want to train MLP or CNN model.")
    parser.add_argument('--model_path', default='',
                        help="you can manually set a .pt model to predict, or the program will automatically retrieve one")
    # todo this argument is only for debug, check if bipolar or unipolar is better.
    parser.add_argument('--bipolar', action='store_true')
    return parser


def find_corresponding_pretrain(args):
    param_path = args.path + args.finetune_folder + '/params_from_preprocess.txt'
    content = read_to_list(param_path)
    target = ''
    for i in range(len(content)):
        if content[i] == "DATA.SUFFIX":
            target = content[i + 1]
            args.feature_string = target
            args.min_size, args.max_size, args.step = read_feature_string(target)
    if args.min_size is None or args.max_size is None or args.step is None:
        raise ValueError("Please check your data preprocessing. The params_from_preprocess.txt was not correctly generated.")

    # find corresponding pretrain data
    training_folders = [f for f in os.listdir(args.path) if
                        os.path.isdir(os.path.join(args.path, f)) and f.startswith("training_data")]
    for folder in training_folders:
        params_path = os.path.join(args.path, folder, "params_from_preprocess.txt")
        content_2 = read_to_list(params_path)
        for i in range(len(content_2)):
            if content_2[i] == "DATA.SUFFIX":
                if content_2[i + 1] == target:
                    return args.path + folder
                else:
                    break

    return ''


def read_to_list(path):
    with open(path, 'r') as file:
        content = file.read()
        return content.split()


def read_feature_string(string):
    parts = string.split('_')
    if len(parts) >= 3:
        try:
            return [int(parts[-3]), int(parts[-2]), int(parts[-1].split('.')[0])]
        except ValueError:
            return None
    else:
        return None


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if not args.treatment:
        args.treatment = input("you didn't assign value from command line, please input a date label for your data here. "
                               "check evaluation/metasurface_verification/predict_params to"
                               "find the right file and specify its date label. e.g. 2024-10-14:\n")
    if not args.finetune_folder:
        args.finetune_folder = input("Path of your finetune data. Required if using predictor under train mode or "
                                     "using matcher. If you didn't specify using commandline, you should add here. "
                                     "e.g. finetune_data_1:\n")

    if args.verify_type == 'matcher':
        args.pretrain_folder = find_corresponding_pretrain(args)
        if args.pretrain_folder == '':
            raise ValueError("For the given finetune data, there is no corresponding pretrain database to do matching.")
        else:
            matcher = Matcher(args)
            JM = matcher.JM_finder()
            waves = matcher.wave
            num_handle = matcher.handled_wave_per_block

    elif args.verify_type == 'predictor':
        predictor = Predictor(args)
        waves = predictor.wave
        num_handle = predictor.handled_wave_per_block
        if args.train:
            predictor.train()
        print("Training process finished. will continue evaluation and prediction...")
        JM = predictor.evaluate()

    else:
        raise ValueError("Invalid verification type! choose from 'matcher' and 'predictor'")

    # for JMs generated from CNN, we need reshape to 2D
    if JM.ndim != 2:
        JM = JM.reshape((JM.shape[0], -1))
    if not isinstance(waves, int):
        num_unit = len(waves) // num_handle
        if (len(waves) % num_handle != 0) and num_unit not in (1, 3, 6):
            raise ValueError(
                "The following code for visualization only handles num_unit in (1, 3, 6), meaning if you want"
                "to consider more num of wavelengths, you can set num_handle to 2, therefore, the available"
                "num_units are (2, 6, 12)")
    else:
        num_unit = 1
    # start the visualization and verfiy logics
    visualizer = Visualizer(args, JM, num_unit, num_handle, waves)
    visualizer.plot()

