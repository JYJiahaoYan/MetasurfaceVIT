
import numpy as np
import json
import os
import re


class Matcher:
    def __init__(self, args):
        self.design_type = args.design_type
        self.treatment = args.treatment
        self.path = args.pretrain_folder
        self.params_path = args.params_path
        self.designJM_path = args.designJM_path
        self.max_size = args.max_size
        self.min_size = args.min_size
        self.step = args.step
        self.feature_string = args.feature_string
        self.angle_step = args.angle_step
        self.wave = None
        self.handled_wave_per_block = None
        self.get_params_from_design()

    def norm_to_real(self):
        norm_param = np.loadtxt(self.params_path + '/type_' + str(self.design_type) + '_' + self.treatment + '.txt')
        # 80 is the maximum of rotate angle, which is fixed.
        maximums = np.array([self.max_size, self.max_size, 80, self.max_size, self.max_size, 80])
        # change to [1, 6] for broadcasting
        maximums = maximums.reshape(1, -1)
        real_param = norm_param * maximums
        return real_param

    def discretize(self, real_param):
        discretized_arr = np.zeros_like(real_param)
        for i in range(6):
            if i in (0, 1, 3, 4):
                discrete_values = np.arange(self.min_size, self.max_size, self.step)
                clipped_arr = np.clip(real_param[i], self.min_size, self.max_size - self.step)
                indices = np.searchsorted(discrete_values, clipped_arr, side='right') - 1
                discretized_arr[i] = discrete_values[indices]
            else:
                discrete_values = np.arange(0, 90, self.angle_step)
                clipped_arr = np.clip(real_param[i], 0, 90 - self.angle_step)
                indices = np.searchsorted(discrete_values, clipped_arr, side='right') - 1
                discretized_arr[i] = discrete_values[indices]
        return discretized_arr

    def index_finder(self):
        """
        this method will load base parameter dataset from corresponding pre-train path, and use the predicted params to
        find the indices that will help the program find corresponding Jones matrices for following steps.
        """
        real_param = self.discretize(self.norm_to_real())
        base_param = np.loadtxt(self.path + '/param_double' + self.feature_string)
        # in each col of param, the first three elements represent one unit, the last three means the other unit.
        # flip these two to produce a new pair of param, which would fully fill the 2D params space, so that you can
        # match with your predicted params better.
        base_param_flip = np.concatenate((base_param[:, 3:6], base_param[:, 0:3]), axis=1)
        matches = np.all(real_param[:, np.newaxis] == base_param, axis=2)
        indices = np.argmax(matches, axis=1)
        no_match = ~np.any(matches, axis=1)
        if np.any(no_match):
            # fixme facing memory overhead issue
            matches_cross = np.all(real_param[no_match][:, np.newaxis] == base_param_flip, axis=2)
            indices_cross = np.argmax(matches_cross, axis=1)
            # negate it to distinguish index from normal or flip base
            indices[no_match] = - indices_cross

        indices[indices == 0] = -1
        return indices

    def JM_finder(self):
        """
        will find and outputcorresponding JMs based on above indices
        """
        JM_all = self.read_pretrain_data()
        indices = self.index_finder()
        # JM_select = np.zeros((len(indices), JM_all.shape[1]), dtype='float16')
        # for i in range(len(indices)):
        #     JM_select[i, :] = JM_all[abs(indices[i]), :]
        JM_select = JM_all[np.abs(indices), :]

        return JM_select

    def read_pretrain_data(self):
        files = [f for f in os.listdir(self.path) if f.startswith('JM_double_No') and f.endswith('.txt')]
        sorted_files = sorted(files, key=lambda f: int(re.search(r'JM_double_No(\d+)', f).group(1)))

        arrays = []
        for file in sorted_files:
            file_path = os.path.join(self.path, file)
            array = np.loadtxt(file_path)
            arrays.append(array)

        combined_array = np.concatenate(arrays, axis=0)

        return combined_array

    def get_params_from_design(self):
        filename = self.designJM_path + '/type_' + str(self.design_type) + '_attr_' + self.treatment + '.json'
        try:
            with open(filename, 'r') as file:
                data = json.load(file)

            self.wave = data.get('wave')
            self.handled_wave_per_block = data.get('handled_wave_per_block')

            print(f"Attributes have been loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}.")
