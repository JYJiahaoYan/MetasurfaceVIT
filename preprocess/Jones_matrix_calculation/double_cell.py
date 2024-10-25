import numpy as np
import os
from .jones_matrix import Element, Rotater
from .visualization import Visualization


class DoubleGenerator:
    def __init__(self, unit_cell, args):
        self.unit_cell = unit_cell
        self.angle_step = args.angle_step
        self.start_wave = args.start_wave
        self.end_wave = args.end_wave
        self.height = args.height
        self.points = args.points
        self.num_of_pieces = args.pieces
        self.feature_string = None
        self.paths = []
        self.root = "./preprocess/"
        self.folder_name = ''
        if args.finetune:
            self.finetune_factor = args.finetune_factor
            # for finetune, data size is much smaller, unnecessary to do data separation
            self.num_of_pieces = 1
        else:
            self.finetune_factor = None
        self.populate_path()
        self.get_wavelengths()
        self.write_unit_info()
        # will put into stats[0]: min; stats[1]: max; stats[2]: mean after traversing all pieces
        # shape will be [3, 6, num_of_points]
        self.stats = np.stack((np.ones((6, self.points)), -1 * np.ones((6, self.points)), np.zeros((6, self.points))), axis=0)

    def populate_path(self):
        # first check the prefix data files, naming as "train_data_1", "train_data_2".
        # if it's up to train_data_n, this newest double_cell object would generate all data in train_data_n+1
        prefix = "finetune_data_" if self.finetune_factor else "training_data_"
        existing_folders = [folder for folder in os.listdir(self.root) if folder.startswith(prefix)]
        if existing_folders:
            existing_numbers = [int(folder.split("_")[-1]) for folder in existing_folders]
            max_number = max(existing_numbers)
            self.folder_name = prefix + str(max_number+1)
        else:
            self.folder_name = prefix + str(1)

        os.makedirs(self.root + self.folder_name, exist_ok=True)

        prefix = 'unit'
        self.feature_string = self.unit_cell.get_path()[len(prefix):]  # e.g. _30_300_10.txt
        self.paths.append(self.root + self.folder_name + '/param_double' + self.feature_string)
        for num in range(self.num_of_pieces):
            save_path = self.root + self.folder_name + '/JM_double_No' + str(num) + self.feature_string
            self.paths.append(save_path)

    def retrieve_unit(self):
        base_param = self.unit_cell.traversed_data()
        name = self.unit_cell.get_path()
        if os.path.exists('./preprocess/FDTD_Simulation/T0_' + name):
            Ax = np.loadtxt('./preprocess/FDTD_Simulation/T0_' + name)
            phix = np.loadtxt('./preprocess/FDTD_Simulation/phase_' + name)
        else:
            self.unit_cell.generate_param_for_FDTD(self.start_wave, self.end_wave, self.points, self.height)
            raise ValueError("No corresponding transmission and phase data were found. Please manually run "
                             "prebuilt.fsp + unit_script.lsf in folder: preprocess/FDTD_Simulation.")
        return base_param, Ax, phix

    def transpose_unit(self):
        base_param, Ax, phix = self.retrieve_unit()
        Ay = np.zeros(np.shape(Ax))
        phiy = np.zeros(np.shape(phix))
        for i in range(len((base_param[:, 0]))):
            for j in range(len((base_param[:, 0]))):
                if (base_param[j, 1] == base_param[i, 2]) and (base_param[j, 2] == base_param[i, 1]):
                    Ay[i] = Ax[j]
                    phiy[i] = phix[j]
        return base_param, Ax, phix, Ay, phiy

    def JM_rotate_unit(self):
        base_param, Ax, phix, Ay, phiy = self.transpose_unit()
        JM = Element(Ax, Ay, phix, phiy)
        base_JM = JM.get_matrix()
        # start adding rotating:
        angles = np.arange(0, 90, self.angle_step)
        unit_para = np.zeros((len(angles), len(base_param[:, 0]), len(base_param[0, :])))
        unit_JM = np.zeros((len(angles), len(base_JM[:, 0, 0, 0]), len(base_JM[0, :, 0, 0]), 2, 2), dtype=complex)
        for i, angle in enumerate(angles):
            JM_left = Rotater(-angle, Ax.shape)
            JM_right = Rotater(angle, Ax.shape)
            unit_para[i, :, :] = np.concatenate((base_param[:, 1:], angle * np.ones((len(base_param), 1))), axis=1)
            # it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
            unit_JM[i, :, :, :] = JM_left.get_matrix() @ base_JM @ JM_right.get_matrix()
        unit_para = unit_para.reshape(-1, len(base_param[0, :]))
        unit_JM = unit_JM.reshape((-1, len(base_JM[0, :, 0, 0]), 2, 2))
        return unit_para, unit_JM

    def JM_double(self):
        unit_para, unit_JM = self.JM_rotate_unit()
        unit_JM = unit_JM.reshape((-1, self.points, 4))
        unit_JM = np.delete(unit_JM, 2, axis=2)
        amount = int((len(unit_para[:, 0]) ** 2 - (len(unit_para[:, 0]))) / 2 + len(unit_para[:, 0]))
        if self.finetune_factor:
            self.JM_double_finetune(unit_para, unit_JM, amount)
        else:
            self.JM_double_pretrained(unit_para, unit_JM, amount)

    def JM_double_finetune(self, unit_para, unit_JM, amount):
        amount_2 = amount // self.finetune_factor
        dimer_para = np.zeros((int(amount_2), 6))
        dimer_JM = np.zeros((int(amount_2), self.points, 3), dtype=complex)
        random_indices = set(np.random.choice(amount, size=amount_2, replace=False))
        count = 0
        for partA in range(len(unit_para[:, 0])):
            for partB in range(partA, len(unit_para[:, 0])):
                if count in random_indices:
                    dimer_para[count, :3], dimer_para[count, 3:] = unit_para[partA, :], unit_para[partB, :]
                    dimer_JM[count, :, :] = unit_JM[partA, :, :] + unit_JM[partB, :, :]
                count += 1
        np.savetxt(self.paths[0], dimer_para, fmt='%d')
        real_JM = self.complex_to_real(dimer_JM, pre_length=0)
        np.savetxt(self.paths[1], real_JM, fmt='%.3f')

    def JM_double_pretrained(self, unit_para, unit_JM, amount):
        dimer_para = np.zeros((int(amount), 6))
        dimer_JM = np.zeros((int(amount), self.points, 3), dtype=complex)
        count = 0
        for partA in range(len(unit_para[:, 0])):
            for partB in range(partA, len(unit_para[:, 0])):
                dimer_para[count, :3], dimer_para[count, 3:] = unit_para[partA, :], unit_para[partB, :]
                dimer_JM[count, :, :] = unit_JM[partA, :, :] + unit_JM[partB, :, :]
                count += 1

        np.savetxt(self.paths[0], dimer_para, fmt='%d')
        batch = len(dimer_JM[:, 0, 0]) // self.num_of_pieces
        pre_length = 0
        for num in range(self.num_of_pieces):
            dynamic_JM = dimer_JM[num * batch:(num + 1) * batch, :, :]
            if num == self.num_of_pieces - 1:
                dynamic_JM = dimer_JM[num * batch:, :, :]
            dimer_train = self.complex_to_real(dynamic_JM, pre_length)
            pre_length = dimer_train.shape[0]
            np.savetxt(self.paths[num + 1], dimer_train, fmt='%.3f')
        # save min max mean stats
        self.stats = self.stats.reshape((-1, self.points))
        np.savetxt(self.root + self.folder_name + '/min_max_mean_list.txt', self.stats, fmt='%.3f')

    def complex_to_real(self, complex_JM, pre_length):
        dimer_amp = np.abs(complex_JM)
        dimer_amp = (dimer_amp - dimer_amp.min()) / (dimer_amp.max() - dimer_amp.min())
        dimer_amp[:, :, 1] = (dimer_amp[:, :, 1] - dimer_amp[:, :, 1].min()) / (
                dimer_amp[:, :, 1].max() - dimer_amp[:, :, 1].min())
        dimer_amp = (dimer_amp - 0.5) / 0.5
        length = dimer_amp.shape[0]
        # fill amps into stats array
        for i in range(3):
            for j in range(self.points):
                self.stats[0, i, j] = min(dimer_amp[:, j, i].min(), self.stats[0, i, j])
                self.stats[1, i, j] = max(dimer_amp[:, j, i].max(), self.stats[1, i, j])
                self.stats[2, i, j] = (dimer_amp[:, j, i].mean() * length + self.stats[2, i, j] * pre_length) / (length + pre_length)
        dimer_phase = np.angle(complex_JM)
        # print('phase confirm min & max', dimer_phase.min(), dimer_phase.max())
        dimer_phase = (dimer_phase - dimer_phase.min()) / (dimer_phase.max() - dimer_phase.min())
        dimer_phase = (dimer_phase - 0.5) / 0.5  # -1 ~ 1
        # print('phase converted min & max', dimer_phase.min(), dimer_phase.max())
        # fill phases into lists
        for i in range(3):
            for j in range(self.points):
                self.stats[0, i + 3, j] = min(dimer_amp[:, j, i].min(), self.stats[0, i + 3, j])
                self.stats[1, i + 3, j] = max(dimer_amp[:, j, i].max(), self.stats[1, i + 3, j])
                self.stats[2, i + 3, j] = (dimer_amp[:, j, i].mean() * length + self.stats[2, i + 3, j] * pre_length) / (length + pre_length)

        real_JM = np.concatenate((dimer_amp, dimer_phase), axis=2).reshape(-1, self.points * 6)
        return real_JM

    def write_lists_to_file(self, list1, list2, list3, filename):
        """
        columns are wavelength 1 - num_of_points
        rows are arranged like this:
        A11_min, A12_min, A22_min, phase11_min, p12_min, p22_min, A11_max, ..., A11_mean ...
        """
        with open(filename, 'w') as file:
            for i in range(0, self.points * 3 * 6, self.points):
                file.write(' '.join(map(str, list1[i:i + self.points])) + '\n')
                file.write(' '.join(map(str, list2[i:i + self.points])) + '\n')
                file.write(' '.join(map(str, list3[i:i + self.points])) + '\n')

    def check_if_exist(self):
        for path in self.paths:
            if not os.path.exists(path):
                return False
        return True

    def visualize_JM(self):
        flag = input('Select if you want to check unit_jm (1) or dimer_jm (2):')
        amount = int(input('input how many random data you want to visualize:'))
        Visualization(self, amount, flag)

    def get_wavelengths(self):
        # use FDTD style to partition wavelengths
        c = 3e8  # m/s speed of light
        wavelengths = [self.start_wave, self.end_wave]  # unit: nm
        # convert to a list (unit: m)
        wavelengths_m = [lambda_ / 1e9 for lambda_ in wavelengths]
        frequencies = [c / lambda_ for lambda_ in wavelengths_m]
        # equally partition frequencies
        frequency_step = (frequencies[1] - frequencies[0]) / (self.points - 1)
        frequencies_list = [frequencies[0] + i * frequency_step for i in range(self.points)]
        waves = [c / frequency * 1e9 for frequency in frequencies_list]
        np.savetxt(self.root + self.folder_name + '/real_wavelengths.txt', waves, fmt='%.3f')
        return waves

    def write_unit_info(self):
        info = [self.unit_cell.max, self.unit_cell.min, self.unit_cell.step]
        np.savetxt(self.root + self.folder_name + '/unit_info.txt', info, fmt='%.3f')

    def generate_list_for_transformer(self):
        suffix = self.feature_string
        pieces = self.num_of_pieces
        points = self.points
        folder_name = self.folder_name + '/'
        info_list = ["DATA.PATH", "./preprocess/", "DATA.FOLDER_NAME", folder_name, "DATA.PREFIX_JM", "JM_double_No",
                     "DATA.PREFIX_PARAM", "param_double", "DATA.SUFFIX", suffix, "DATA.DIVIDE_NUM", pieces, "DATA.SIZE_X", points]
        with open(self.root + self.folder_name + '/params_from_preprocess.txt', 'w') as file:
            file.write(' '.join(map(str, info_list)))






