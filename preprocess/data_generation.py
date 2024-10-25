import argparse
from Jones_matrix_calculation.double_cell import DoubleGenerator
from FDTD_Simulation.unit_cell import UnitGenerator


# preprocess part uses the following local args_parse in order not to mess up config.py for transformer part.
def get_args_parser():
    parser = argparse.ArgumentParser('MetasurfaceVIT-preprocess', add_help=False)
    # for unit
    parser.add_argument('--min_size', default=30, type=int,
                        help='The minimum x and y length of designed silicon nano-blocks (unit: nm)')
    parser.add_argument('--max_size', default=300, type=int,
                        help='The maximum x and y length of designed silicon nano-blocks (unit: nm)')
    parser.add_argument('--step', default=10, type=int,
                        help='The interval between adjacent size points (unit: nm)')
    # for double
    parser.add_argument('--angle_step', default=10, type=int, help='The interval between adjacent angle points '
                                                                   '(unit: deg). notice the min & max of angles '
                                                                   'are fixed to 0 and 80 deg')
    parser.add_argument('--start_wave', default=400, type=int,
                        help='The lower limit wavelength of incident light (unit: nm)')
    parser.add_argument('--end_wave', default=800, type=int,
                        help='The upper limit wavelength of incident light (unit: nm)')
    parser.add_argument('--points', default=20, type=int,
                        help='Number of wavelength points from start_wave to end_wave, rendering a points x 6 Jones '
                             'Matrix for following transformer training')
    parser.add_argument('--height', default=450, type=int,
                        help="Unit: nm. it is an invariant during training. please set to proper value or use default.")
    parser.add_argument('--pieces', default=10, type=int,
                        help='the number of pieces (as txt file) you planned to divide generated Jones matrix data and save locally.'
                             'Since this project is planned to run on majority of computers, saving files separately '
                             'decreases the possibility to reach your memory limitation.')
    parser.add_argument('--visualize', default=False, type=bool,
                        help='whether you choose to visualize the generated Jones matrices of unit or dimer')
    parser.add_argument("--finetune", action='store_true', help="add it if you want generate lightweighted data for finetuning")
    parser.add_argument("--finetune_factor", default=10, type=int,
                        help=' how many times smaller the fine-tuning dataset is compared to the pre-training dataset'
                             'could be /10  /20 or /30...')

    return parser


def main(args):
    unit = UnitGenerator(args)
    double = DoubleGenerator(unit, args)
    double.generate_list_for_transformer()
    if double.check_if_exist():
        raise RuntimeError('All necessary data for training already exist. Don t need to run preprocess!')
    else:
        double.JM_double()
    if args.visualize:
        double.visualize_JM()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
