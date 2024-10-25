import csv
import numpy as np
import os


class UnitGenerator:
    def __init__(self, args):
        self.max = args.max_size
        self.min = args.min_size
        self.step = args.step
        self.path = 'unit_' + str(self.min) + '_' + str(self.max) + '_' + str(self.step) + '.txt'  # e.g., unit_30_300_10.txt
        self.count = None

    def traversed_data(self):
        if os.path.exists(self.path):
            data = np.loadtxt()
        else:
            x_all = []
            y_all = []
            count_all = []
            count = 1

            for x in range(self.min, self.max, self.step):
                for y in range(self.min, self.max, self.step):
                    x_all.append(x)
                    y_all.append(y)
                    count_all.append(count)
                    count += 1

            count_all = np.array(count_all, ndmin=2).transpose()
            self.count = len(count_all)
            x_all = np.array(x_all, ndmin=2).transpose()
            y_all = np.array(y_all, ndmin=2).transpose()
            data = np.concatenate((count_all, x_all, y_all), axis=1)
            np.savetxt('./preprocess/FDTD_Simulation/' + self.path, data, fmt='%d')
        return data

    def generate_param_for_FDTD(self, start_wave: int=400, end_wave: int=800, points: int=20, height: int=450):
        # count, start wavelength, stop wavelength,frequency points, height, new_path
        # new path is a string that will be used by fdtd as output file name
        params = [self.count, start_wave, end_wave, points, height, self.path]
        with open('./preprocess/FDTD_Simulation/param_for_FDTD.txt', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(params)

    def get_path(self):
        return self.path


