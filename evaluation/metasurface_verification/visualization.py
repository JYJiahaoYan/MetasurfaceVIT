import numpy as np
import matplotlib.pyplot as plt
import os


class Visualizer:
    def __init__(self, args, JM, num_unit, num_handle, waves):
        self.num_unit = num_unit
        self.JM = JM
        self.num_handle = num_handle
        # num_handle * num_unit == num_waves.
        # self.size can be regarded as unit size
        # if num_unit = 3, the whole metasurface would arrange as [3*size, size]
        # if num_unit = 6, the whole metasurface would arrange as [3*size, 2*size]
        self.size = int(np.sqrt(JM.shape[0] // (num_handle * num_unit)))
        self.info_string = args.treatment
        # will be 'predictor' or 'matcher', combing with info_string to form a file name.
        self.source = args.verify_type
        self.design_type = args.design_type
        self.waves = waves
        self.value_validate()
        if args.verify_type == "matcher":
            self.select_wavelength()

    def value_validate(self):
        print(f'please check JM.max({self.JM.max()}): and JM.min({self.JM.min()}), which should be roughly located in 0-1')
        print(f'shape of JM:{np.shape(self.JM)}')
        if self.JM.ndim != 2:
            raise Exception("THe imported Jones Matrices' shapes must be [nums, num_wave * 6].")
        # for RGB design, there are some limitations
        if self.design_type == 3 and (self.num_unit == 6 or (self.num_unit == 1 and self.num_handle != 3) or
                                      (self.num_unit == 3 and self.num_handle != 1)):
            raise Exception("For design_type3(RGB design), available num_unit and num_handle pairs are (1,3) and (3,1)")

    def select_wavelength(self):
        self.JM = self.JM.reshape((self.JM.shape[0], -1, 6))
        selected = self.JM[:, self.waves, :]
        self.JM = selected

    def plot(self):
        # first, scale the amplitude values and phase values back to their actual values
        min_amp = 0
        max_amp = 1.982
        min_phase = -np.pi
        max_phase = np.pi
        for i in range(self.JM.shape[1]):
            if i % 6 < 3:  # amplitude part
                self.JM[:, i] = self.JM[:, i] * (max_amp - min_amp) + min_amp
            else:  # phase part
                self.JM[:, i] = self.JM[:, i] * (max_phase - min_phase) + min_phase

        if self.num_unit == 1:
            self.plot_1()
        elif self.num_unit == 3:
            self.plot_3()
        elif self.num_unit == 6:
            self.plot_6()
        else:
            raise ValueError("Invalid num_unit argument. Can only be in (1, 3, 6)")

    def plot_1(self):
        if self.design_type == 3:
            size = int(np.sqrt(self.JM.shape[0]))
            self.JM = self.JM.reshape((size, size, -1))
            self.process_visualize_rgb(self.JM)
        else:
            # package into amp list and phase list
            amps = [self.JM[:, 0], self.JM[:, 1], self.JM[:, 2]]
            phases = [self.JM[:, 3], self.JM[:, 4], self.JM[:, 5]]
            self.process_visualize(amps, phases)

    def plot_3(self):
        """
        3 blocks should arrange like this:
        double1 (unitA unitB)
        double2 (unitC unitD)
        double3 (unitE unitF)
        if you don't familiar with above terminology, please check files in: preprocess/Jones_matrix_calculation
        """
        if self.size * self.size * 3 == self.JM.shape[0]:
            # new array will be used to unpack JM arrays for different blocks, and arrange them separately in space
            new_array = np.zeros((self.size * 3, self.size, self.JM.shape[1]))
            print('shape of new_array:', np.shape(new_array))
        else:
            raise ValueError(f"JM_select_unit.shape[0] ({self.JM.shape[0]}) != size*size*3 ({self.size * self.size * 3})")
        for row in range(new_array.shape[0]):
            for col in range(new_array.shape[1]):
                large_row = row // 3
                large_col = col
                if row % 3 == 0:
                    unit = 1
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 1:
                    unit = 2
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 2:
                    unit = 3
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
        if self.design_type == 3:
            self.process_visualize_rgb(new_array, ratio=3)
        else:
            for num in range(self.num_unit):
                amps = [new_array[:, :, 0 + 6 * num], new_array[:, :, 1 + 6 * num], new_array[:, :, 2 + 6 * num]]
                phases = [new_array[:, :, 3 + 6 * num], new_array[:, :, 4 + 6 * num], new_array[:, :, 5 + 6 * num]]
                self.process_visualize(amps, phases, ratio=3)

    def plot_6(self):
        """
        6 blocks should arrange like this:
        double1 (unitA unitB)  double2 (unitC unitD)
        double3 (unitE unitF)  double4 (unitG unitH)
        double5 (unitI unitJ)  double6 (unitK unitL)
        if you don't familiar with above terminology, please check files in: preprocess/Jones_matrix_calculation
        """
        if self.size * self.size * 6 == self.JM.shape[0]:
            new_array = np.zeros((self.size * 3, self.size * 2, self.JM.shape[1]))
        else:
            raise ValueError(f"JM_select_unit.shape[0] ({self.JM.shape[0]}) != size * 6 ({self.size * 6})")
        for row in range(new_array.shape[0]):
            for col in range(new_array.shape[1]):
                large_row = row // 3
                large_col = col // 2
                if row % 3 == 0 and col % 2 == 0:
                    unit = 1
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    # if we regard 3x2 group as a unit, their locations:
                    # former array C-like index order,
                    # with the last axis index changing fastest, back to the first axis index changing slowest.
                    # that is ,large col changes faster
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 1 and col % 2 == 0:
                    unit = 3
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 2 and col % 2 == 0:
                    unit = 5
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 0 and col % 2 == 1:
                    unit = 2
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 1 and col % 2 == 1:
                    unit = 4
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
                elif row % 3 == 2 and col % 2 == 1:
                    unit = 6
                    locate_region = self.JM[(unit - 1) * self.size * self.size: unit * self.size * self.size, :]
                    new_array[row, col, :] = locate_region[self.size * large_row + large_col, :]
        for num in range(self.num_unit):
            amps = [new_array[:, :, 0 + 6 * num], new_array[:, :, 1 + 6 * num], new_array[:, :, 2 + 6 * num]]
            phases = [new_array[:, :, 3 + 6 * num], new_array[:, :, 4 + 6 * num], new_array[:, :, 5 + 6 * num]]
            self.process_visualize(amps, phases, ratio=6)

    def process_visualize_rgb(self, JM, ratio=1):
        # todo still needs to verify two situations: unit1+handle3 & unit3+handle1
        # the second dimension is like A11_R, A12_R, A22_R, phi11_R, phi12_R, phi22_R, ...., A11_B, A12_B, A22_B, phi11_B, phi12_B, phi22_B
        amps2 = []
        phases2 = []
        # loop for three polarization states
        for i in range(3):
            temp = np.stack((JM[:, :, i * 6], JM[:, :, i * 6 + 1], JM[:, :, i * 6 + 2]), axis=2)
            amps2.append(temp)
            temp = np.stack((JM[:, :, i * 6 + 3], JM[:, :, i * 6 + 4], JM[:, :, i * 6 + 5]), axis=2)
            phases2.append(temp)

        for i in range(len(amps2)):
            amp = amps2[i].reshape((self.size * ratio, self.size, 3))
            # change to 0-1/ then change amplitude to light intensity/ then change from linear RGB to standardRGB (sRGB)
            amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
            amp = amp ** 2
            amp = linear_to_srgb(amp)
            phase = phases2[i].reshape((self.size * ratio, self.size, 3))
            holo_int = self.Holo_Calculation(amp, phase)
            # clip into 0-1
            holo_int = np.clip(holo_int, a_min=None, a_max=1)
            rgb_plot(amp)
            rgb_plot(holo_int)

    def process_visualize(self, amps, phases, ratio=1):
        for i in range(len(amps)):
            amps[i] = amps[i].reshape((self.size * ratio, self.size))
            # change to 0-1/ then change amplitude to light intensity/ then change from linear RGB to standardRGB (sRGB)
            amps[i] = (amps[i] - amps[i].min()) / (amps[i].max() - amps[i].min() + 1e-8)
            amps[i] = amps[i] ** 2
            amps[i] = linear_to_srgb(amps[i])
            array_plot(amps[i], "Printing Intensity Distribution")
            phases[i] = phases[i].reshape((self.size * ratio, self.size))
            holo_int = self.Holo_Calculation(amps[i], phases[i])
            array_plot(holo_int, "Holographic Intensity Distribution")

    def Holo_Calculation(self, amp, phase):
        wave_plate = amp * np.exp(1j * phase)
        if self.design_type == 3:
            F = np.fft.fft2(wave_plate, axes=(0, 1), norm="ortho")
        else:
            F = np.fft.fft2(wave_plate, norm="ortho")
        F_ = np.fft.fftshift(F)
        holo_intensity = abs(F_) ** 2
        return holo_intensity


"""
utils functions shown below
"""


def srgb_to_linear(c_srgb):
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear


def linear_to_srgb(c_linear):
    c_srgb = np.where(c_linear <= 0.0031308, 12.92 * c_linear, 1.055 * (c_linear ** (1/2.4)) - 0.055)
    return c_srgb


def array_plot(im, name):
    x = np.arange(int(-im.shape[1] / 2), int(im.shape[1] / 2))
    y = np.arange(int(-im.shape[0] / 2), int(im.shape[0] / 2))
    plt.title(name)
    plt.pcolormesh(x, y, im)
    plt.set_cmap('gray')
    plt.xlim(int(-im.shape[1] / 2), int(im.shape[1] / 2) - 1)
    plt.ylim(int(-im.shape[0] / 2), int(im.shape[0] / 2) - 1)
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()


def rgb_plot(im):
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("Input image should have 3 dimensions with 3 channels (RGB)")

    x = np.arange(int(-im.shape[1] / 2), int(im.shape[1] / 2))
    y = np.arange(int(-im.shape[0] / 2), int(im.shape[0] / 2))

    plt.figure()
    plt.title('Calculated RGB Intensity')

    plt.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')

    plt.xlim(int(-im.shape[1] / 2), int(im.shape[1] / 2) - 1)
    plt.ylim(int(-im.shape[0] / 2), int(im.shape[0] / 2) - 1)

    plt.colorbar(label='Intensity')
    plt.clim(0, 1)
    plt.show()

