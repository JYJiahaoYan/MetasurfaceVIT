import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, double_cell, amount, flag):
        self.double_cell = double_cell
        self.amount = amount
        self.wavelengths = self.double_cell.get_wavelengths()
        if flag == '1':
            self.visualize_unit()
        elif flag == '2':
            self.visualize_double()
        else:
            raise ValueError('input must be 1 or 2')

    def visualize_unit(self):
        # retrieve JM of rotated unit
        unit_para, unit_JM = self.double_cell.JM_rotate_unit()  # e.g., unit_jm size [6561, 20, 2, 2]
        numbers = np.random.randint(low=0, high=len(unit_para), size=self.amount)

        for num in numbers:
            chosen_para = unit_para[num, :]
            chosen_para = 'No.' + str(num) + ':x' + str(chosen_para[0]) + 'y' + str(chosen_para[1]) + 'angle' + str(chosen_para[2])
            # ignore J21 cause it's the same as J12
            chosen_J11 = unit_JM[num, :, 0, 0]
            J11_amp, J11_phase = np.abs(chosen_J11), np.angle(chosen_J11)
            chosen_J12 = unit_JM[num, :, 0, 1]
            J12_amp, J12_phase = np.abs(chosen_J12), np.angle(chosen_J12)
            chosen_J22 = unit_JM[num, :, 1, 1]
            J22_amp, J22_phase = np.abs(chosen_J22), np.angle(chosen_J22)
            drawing_list = [J11_amp, J12_amp, J22_amp, J11_phase, J12_phase, J22_phase]
            subtitles = ['J11_amp', 'J12_amp', 'J22_amp', 'J11_phase', 'J12_phase', 'J22_phase']
            self.drawing(chosen_para, drawing_list, subtitles)

    def visualize_double(self):
        # retrieve JM from file
        files = self.double_cell.paths
        random1 = np.random.randint(1, len(files))
        chosen_file = files[random1]
        double_para = np.loadtxt(files[0])
        double_JM = np.loadtxt(chosen_file).reshape((-1, self.double_cell.points, 6))
        batch = len(double_JM)
        double_para = double_para[(random1 - 1) * batch : random1 * batch]
        numbers = np.random.randint(low=0, high=len(double_para), size=self.amount)

        for num in numbers:
            chosen_para = double_para[num, :]
            chosen_para = 'No.' + str(num) + ':x1:' + str(chosen_para[0]) + 'y1:' + str(chosen_para[1]) + 'angle1:' + \
                          str(chosen_para[2]) + 'x2:' + str(chosen_para[3]) + 'y2:' + str(chosen_para[4]) + 'angle2:' \
                          + str(chosen_para[5])
            J11_amp, J12_amp, J22_amp = double_JM[num, :, 0], double_JM[num, :, 1], double_JM[num, :, 2]
            J11_phase, J12_phase, J22_phase = double_JM[num, :, 3], double_JM[num, :, 4], double_JM[num, :, 5]
            drawing_list = [J11_amp, J12_amp, J22_amp, J11_phase, J12_phase, J22_phase]
            subtitles = ['J11_amp', 'J12_amp', 'J22_amp', 'J11_phase', 'J12_phase', 'J22_phase']
            self.drawing(chosen_para, drawing_list, subtitles)

    def drawing(self, chosen_para, drawing_list, subtitles):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(chosen_para, fontsize=16)
        for i in range(len(drawing_list)):
            # subplots
            plt.subplot(2, 3, i+1)  # J11 amp
            plt.plot(self.wavelengths, drawing_list[i])
            plt.title(subtitles[i])

        plt.tight_layout()
        plt.show()
