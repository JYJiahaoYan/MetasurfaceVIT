import datetime
import numpy as np
from utils import GSCalculation, LensPhase
import json


class JMGenerator:
    def __init__(self, args, params, imgs, norm_imgs):
        self.params = params
        self.pretrain_path = args.pretrain_path
        self.output_path = args.output_path
        self.design_type = args.design_type
        self.visualize = args.visualize
        self.random_seed = args.random_seed
        self.bias = args.bias
        self.noise = args.noise
        self.size = args.size
        self.wave = args.fixed_wave
        self.num_waves = int(self.params['DATA.SIZE_X'])
        self.amplitude = args.amplitude
        self.handled_wave_per_block = args.handled_wave_per_block
        self.imgs = imgs  # 0 - 255 or 0 - 1 intensity
        self.norm_imgs = norm_imgs  # -1 to 1 amplitude
        self.tolerance = args.tolerance
        self.focus = args.focus_length
        self.stats = None
        self.stats_from_pretrained_JM()
        if self.design_type != 4:
            assert self.imgs[0].shape[0] == self.size, "Size mismatch! Please check your image generator."
        if not self.wave and self.design_type in (1, 2):
            self.populate_wavelength()

    def populate_wavelength(self):
        """
        if design type is 1 (2), randomly choose 1 (3) wavelength points from num_waves
        """
        if self.design_type == 1:
            self.wave = np.random.randint(self.num_waves)
        else:
            self.wave = np.random.randint(self.num_waves, size=3)

    def JM_type1(self):
        A11_map = self.norm_imgs[0]
        A22_map = self.norm_imgs[1]
        A12_map = self.norm_imgs[2]
        A11_map = self.bias_adjustment(A11_map, self.wave, 0)
        A22_map = self.bias_adjustment(A22_map, self.wave, 2)
        A12_map = self.bias_adjustment(A12_map, self.wave, 1)
        if self.noise:
            A11_map, A22_map, A12_map = self.add_noise(A11_map), self.add_noise(A22_map), self.add_noise(A12_map)
        phi11_map = GSCalculation(self.back_01intensity(A11_map), self.imgs[3], self.tolerance, self.visualize,
                                  self.phase_alignment(self.wave, 0))
        phi22_map = GSCalculation(self.back_01intensity(A22_map), self.imgs[4], self.tolerance, self.visualize,
                                  self.phase_alignment(self.wave, 2))
        phi12_map = GSCalculation(self.back_01intensity(A12_map), self.imgs[5], self.tolerance, self.visualize,
                                  self.phase_alignment(self.wave, 1))
        # change from -pi to pi to -1 to 1
        phi11_map = phi11_map / np.pi
        phi22_map = phi22_map / np.pi
        phi12_map = phi12_map / np.pi

        JM_partial = np.stack((A11_map, A12_map, A22_map, phi11_map, phi12_map, phi22_map), axis=2)
        # return array [sizex, sizey, 6] will need to merge into JM_all [sizex, sizey, num_wave, 6]
        return JM_partial

    def JM_type2(self):
        if len(self.wave) % self.handled_wave_per_block != 0:
            raise ValueError("Your setting leads to a non-integer block number!")
        # notice: one block means a dimer structure containing two units. These dimer structures are basis to form our database
        JM_partial = []
        count = 0
        for wavelength in self.wave:
            A11_map = self.norm_imgs[0 + 6 * count]
            A22_map = self.norm_imgs[1 + 6 * count]
            A12_map = self.norm_imgs[2 + 6 * count]
            A11_map = self.bias_adjustment(A11_map, wavelength, 0)
            A22_map = self.bias_adjustment(A22_map, wavelength, 2)
            A12_map = self.bias_adjustment(A12_map, wavelength, 1)
            if self.noise:
                A11_map, A22_map, A12_map = self.add_noise(A11_map), self.add_noise(A22_map), self.add_noise(A12_map)
            phi11_map = GSCalculation(self.back_01intensity(A11_map), self.imgs[3 + 6 * count], self.tolerance,
                                      self.visualize, self.phase_alignment(wavelength, 0))
            phi22_map = GSCalculation(self.back_01intensity(A22_map), self.imgs[4 + 6 * count], self.tolerance,
                                      self.visualize, self.phase_alignment(wavelength, 2))
            phi12_map = GSCalculation(self.back_01intensity(A12_map), self.imgs[5 + 6 * count], self.tolerance,
                                      self.visualize, self.phase_alignment(wavelength, 1))
            phi11_map = phi11_map / np.pi
            phi22_map = phi22_map / np.pi
            phi12_map = phi12_map / np.pi
            JM_item = np.stack((A11_map, A12_map, A22_map, phi11_map, phi12_map, phi22_map), axis=2)
            JM_partial.append(JM_item)

        return JM_partial

    def JM_type3(self):
        # different from JM_type1 and type2, here input images are colorful
        wavelength_path = self.pretrain_path + '/real_wavelengths.txt'
        wavelengths = np.loadtxt(wavelength_path)
        rgb_wavelengths = {
            'R': 620,
            'G': 540,
            'B': 450
        }
        closest = [min(enumerate(wavelengths), key=lambda x: abs(x[1] - rgb_wavelengths[color])) for color in 'RGB']
        self.wave = [x[0] for x in closest]
        closest_waves = [x[1] for x in closest]
        print(f"The selected wavelength channels are {self.wave}, which are the closest to RGB channels.")
        A11_map_R = self.bias_adjustment(self.norm_imgs[0][:, :, 0], self.wave[0], 0)
        A22_map_R = self.bias_adjustment(self.norm_imgs[1][:, :, 0], self.wave[0], 2)
        A12_map_R = self.bias_adjustment(self.norm_imgs[2][:, :, 0], self.wave[0], 1)
        A11_map_G = self.bias_adjustment(self.norm_imgs[0][:, :, 1], self.wave[1], 0)
        A22_map_G = self.bias_adjustment(self.norm_imgs[1][:, :, 1], self.wave[1], 2)
        A12_map_G = self.bias_adjustment(self.norm_imgs[2][:, :, 1], self.wave[1], 1)
        A11_map_B = self.bias_adjustment(self.norm_imgs[0][:, :, 2], self.wave[2], 0)
        A22_map_B = self.bias_adjustment(self.norm_imgs[1][:, :, 2], self.wave[2], 2)
        A12_map_B = self.bias_adjustment(self.norm_imgs[2][:, :, 2], self.wave[2], 1)
        if self.noise:
            A11_map_R, A22_map_R, A12_map_R = self.add_noise(A11_map_R), self.add_noise(A22_map_R), self.add_noise(A12_map_R)
            A11_map_G, A22_map_G, A12_map_G = self.add_noise(A11_map_G), self.add_noise(A22_map_G), self.add_noise(A12_map_G)
            A11_map_B, A22_map_B, A12_map_B = self.add_noise(A11_map_B), self.add_noise(A22_map_B), self.add_noise(A12_map_B)
        # for hologram, these three RGB channels also work together to present a colorful image
        phi11_map_R = GSCalculation(self.back_01intensity(A11_map_R), self.imgs[3][:, :, 0], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[0], 0))
        phi22_map_R = GSCalculation(self.back_01intensity(A22_map_R), self.imgs[4][:, :, 0], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[0], 2))
        phi12_map_R = GSCalculation(self.back_01intensity(A12_map_R), self.imgs[5][:, :, 0], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[0], 1))
        phi11_map_R = phi11_map_R / np.pi
        phi22_map_R = phi22_map_R / np.pi
        phi12_map_R = phi12_map_R / np.pi
        phi11_map_G = GSCalculation(self.back_01intensity(A11_map_G), self.imgs[3][:, :, 1], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[1], 0))
        phi22_map_G = GSCalculation(self.back_01intensity(A22_map_G), self.imgs[4][:, :, 1], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[1], 2))
        phi12_map_G = GSCalculation(self.back_01intensity(A12_map_G), self.imgs[5][:, :, 1], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[1], 1))
        phi11_map_G = phi11_map_G / np.pi
        phi22_map_G = phi22_map_G / np.pi
        phi12_map_G = phi12_map_G / np.pi
        phi11_map_B = GSCalculation(self.back_01intensity(A11_map_B), self.imgs[3][:, :, 2], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[2], 0))
        phi22_map_B = GSCalculation(self.back_01intensity(A22_map_B), self.imgs[4][:, :, 2], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[2], 2))
        phi12_map_B = GSCalculation(self.back_01intensity(A12_map_B), self.imgs[5][:, :, 2], self.tolerance,
                                    self.visualize, self.phase_alignment(self.wave[2], 1))
        phi11_map_B = phi11_map_B / np.pi
        phi22_map_B = phi22_map_B / np.pi
        phi12_map_B = phi12_map_B / np.pi

        JM_R = np.stack((A11_map_R, A12_map_R, A22_map_R, phi11_map_R, phi12_map_R, phi22_map_R), axis=2)
        JM_G = np.stack((A11_map_G, A12_map_G, A22_map_G, phi11_map_G, phi12_map_G, phi22_map_G), axis=2)
        JM_B = np.stack((A11_map_B, A12_map_B, A22_map_B, phi11_map_B, phi12_map_B, phi22_map_B), axis=2)

        return [JM_R, JM_G, JM_B]

    # notice, instead of partial JM, here we directly get the whole JM since it's a broadband metalens.
    def JM_type4(self):
        wavelength_path = self.pretrain_path + '/real_wavelengths.txt'
        wavelengths = np.loadtxt(wavelength_path)
        JM_in_Lens = np.ones((self.size, self.size, self.num_waves, 6))
        # adjust amplitudes
        for i in range(self.num_waves):
            for j in range(3):
                JM_in_Lens[:, :, i, j] = self.bias_adjustment(JM_in_Lens[:, :, i, j], i, j)
        waves = np.array(wavelengths) * 1e-9
        # notice unit is nm. generate pitch from max size of unit cell. will be max + 100nm
        # pitch x will be the double as pitch y, because a block contains two units placing along x-axis
        unit_info = np.loadtxt(self.pretrain_path + '/unit_info.txt')
        unitx = 2 * (unit_info[0] + 100) * 1e-9
        unity = (unit_info[0] + 100) * 1e-9
        focus_length = self.focus * 1e-6
        if self.visualize:
            visual = int(input(f"Choose a wavelength point from 0 to {self.num_waves - 1} for visualization the phase "
                               f"distribution. enter an integer larger than this range if you don't want to visualize."))
        else:
            visual = self.num_waves + 100
        flag = False if visual >= self.num_waves else True
        for index, wave in enumerate(waves):
            if index > 0:
                flag = False
            JM_in_Lens[:, :, index, 3] = LensPhase(self.size, self.size, unitx, unity, focus_length, 0, 0, wave,
                                                   flag) / np.pi
            JM_in_Lens[:, :, index, 4] = LensPhase(self.size, self.size, unitx, unity, focus_length, 0, 0, wave,
                                                   flag) / np.pi
            JM_in_Lens[:, :, index, 5] = LensPhase(self.size, self.size, unitx, unity, focus_length, 0, 0, wave,
                                                   flag) / np.pi

        return JM_in_Lens

    def phase_alignment(self, wavelength, subscript):
        # only cares about aligning the mean values
        return self.stats[2, subscript + 3, wavelength]

    def stats_from_pretrained_JM(self):
        path = 'preprocess/' + self.params['DATA.FOLDER_NAME'] + '/min_max_mean_list.txt'
        stats = np.loadtxt(path).reshape((3, 6, self.num_waves))
        # e.g., stats[0, :, :] is min_related stats
        # stats[1, 0, :] is max_A11 at varied wavelengths
        self.stats = stats

    def bias_adjustment(self, amp_matrix, wavelength, subscript):
        _min, _max, _mean = self.stats[0, subscript, wavelength], self.stats[1, subscript, wavelength], self.stats[2, subscript, wavelength]
        shrink_min = _mean + (_min - _mean) * (1 - self.bias)
        shrink_max = _mean + (_max - _mean) * (1 - self.bias)
        amp_matrix = self.scale_data(amp_matrix, shrink_min, shrink_max)
        return amp_matrix

    def scale_data(self, data, new_min, new_max, old_min=-1, old_max=1):
        normalized_data = (data - old_min) / (old_max - old_min)
        scaled_data = normalized_data * (new_max - new_min) + new_min
        return scaled_data

    def add_noise(self, matrix):
        random = np.random.uniform(-1 * self.noise, self.noise, size=np.shape(matrix))
        matrix += random
        return matrix

    def back_01intensity(self, amp):
        amp01 = 0.5 * amp + 0.5
        return amp01 ** 2

    # wrap up functions are aimed to populate vacancies in Jones Matrices and then generate corresponding masks
    # notice mask == 0 means preserved part, while 1 means masked part.
    def JM_type1_wrapup(self):
        designed_JM = self.JM_type1()
        JM = -1 * np.ones((self.size, self.size, self.num_waves, 6))
        mask = np.ones((self.size, self.size, self.num_waves, 6))
        for i in range(self.num_waves):
            for j in range(3):
                if i == self.wave:
                    JM[:, :, i, :] = designed_JM[i]
                else:
                    JM[:, :, i, j] = self.bias_adjustment(JM[:, :, i, j], i, j)
        if self.amplitude == 'one':
            mask[:, :, self.wave, :] = 0
        elif self.amplitude == 'all':
            mask[:, :, :, 0:3] = 0
            mask[:, :, self.wave, 3:6] = 0

        return JM, mask

    def JM_type2_wrapup(self):
        num_block = len(self.wave) // self.handled_wave_per_block
        step = self.handled_wave_per_block
        designed_JMs = self.JM_type2()  # length should be len(self.waves)
        JM = -1 * np.ones((self.size, self.size, self.num_waves, 6))
        count = 0
        for i in range(self.num_waves):
            if i in self.wave:
                JM[:, :, i, :] = designed_JMs[count]
                count += 1
            else:
                for j in range(3):
                    JM[:, :, i, j] = self.bias_adjustment(JM[:, :, i, j], i, j)
        mask = np.ones((self.size, self.size, self.num_waves, 6))
        masks = []
        if self.amplitude == 'one':
            start = 0
            end = step
            for i in range(num_block):
                curr_mask = mask.copy()
                curr_mask[:, :, self.wave[start:end], :] = 0
                masks.append(curr_mask)
                start += step
                end += step
        elif self.amplitude == 'all':
            # preserve all amplitudes, but only preserve the phase of targeted wavelength
            start = 0
            end = step
            for i in range(num_block):
                curr_mask = mask.copy()
                curr_mask[:, :, :, 0:3] = 0
                curr_mask[:, :, self.wave[start:end], 3:6] = 0
                masks.append(curr_mask)
                start += step
                end += step
        else:  # several
            start = 0
            end = step
            for i in range(num_block):
                curr_mask = mask.copy()
                curr_mask[:, :, self.wave, 0:3] = 0
                curr_mask[:, :, self.wave[start:end], 3:6] = 0
                masks.append(curr_mask)
                start += step
                end += step

        return JM, masks

    def JM_type3_wrapup(self):
        designed_JMs = self.JM_type3()
        # first, bias_adjustment of background JM. Change the originally designed -1 (min) to a loose min
        JM = -1 * np.ones((self.size, self.size, self.num_waves, 6))
        # adjust amplitudes  (we don't care about phases, which will be covered by mask)
        count = 0
        for i in range(self.num_waves):
            if i in self.wave:
                JM[:, :, i, :] = designed_JMs[count]
                count += 1
            else:
                for j in range(3):
                    JM[:, :, i, j] = self.bias_adjustment(JM[:, :, i, j], i, j)
        # generate mask based on different amplitude-masking strategies
        assert self.handled_wave_per_block == 1, "handled_wave_per_block must be one for this RGB application design!"
        mask = np.ones((self.size, self.size, self.num_waves, 6))
        masks = [mask.copy(), mask.copy(), mask.copy()]
        if self.amplitude == 'one':
            for i, w in enumerate(self.wave):
                masks[i][:, :, w, :] = 0
        elif self.amplitude == 'all':
            # preserve all amplitudes, but only preserve the phase of targeted wavelength
            for i, w in enumerate(self.wave):
                masks[i][:, :, :, 0:3] = 0
                masks[i][:, :, w, 3:6] = 0
        else:  # several
            for i, w in enumerate(self.wave):
                masks[i][:, :, self.wave, 0:3] = 0
                masks[i][:, :, w, 3:6] = 0
        # notice JM is identical for different blocks, but we use masks to make differences
        return JM, masks

    def JM_type4_wrapup(self):
        JM = self.JM_type4()
        if self.amplitude == 'all':
            mask = np.zeros((self.size, self.size, self.num_waves, 6))
        else:  # 'none'
            mask = np.concatenate((np.ones((self.size, self.size, self.num_waves, 3)),
                                   np.zeros((self.size, self.size, self.num_waves, 3))), axis=3)
        # JM = JM.reshape((self.size * self.size, self.num_waves, 6))
        # mask = mask.reshape((self.size * self.size, self.num_waves, 6))
        return JM, mask

    def generate_and_save(self):
        if self.design_type == 1:
            JM, mask = self.JM_type1_wrapup()
        elif self.design_type == 2:
            JM, mask = self.JM_type2_wrapup()
        elif self.design_type == 3:
            JM, mask = self.JM_type3_wrapup()
        elif self.design_type == 4:
            JM, mask = self.JM_type4_wrapup()
        else:
            raise ValueError("invalid design type!")
        today = datetime.date.today()
        date = today.strftime('%Y-%m-%d')
        if isinstance(mask, list):
            # create a new 0-axis to stack different masks
            mask = np.stack(mask, axis=0)
        np.save(self.output_path + '/type_' + str(self.design_type) + '_mask_' + date + '.npy', mask)
        np.save(self.output_path + '/type_' + str(self.design_type) + '_JM_' + date + '.npy', JM)
        self.save_attributes_to_file(self.output_path + '/type_' + str(self.design_type) + '_attr_' + date + '.json')

    def save_attributes_to_file(self, filename):
        data = {
            "wave": self.wave,
            "handled_wave_per_block": self.handled_wave_per_block
        }
        data = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in data.items()}
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Attributes have been saved to {filename}")
