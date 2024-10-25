import numpy as np


class JonesMatrix:
    def __init__(self, matrix):
        if isinstance(matrix, np.ndarray) and matrix[0,0,:,:].shape == (2, 2):
            if matrix.ndim == 4:
                self.matrix = np.array(matrix).astype(complex)
            else:
                raise ValueError('Shape of array/matrix must be index x lambda x 2 x 2')
        else:
            raise ValueError('Shape of array/matrix must be index x lambda x 2 x 2')

    def __repr__(self):
        return 'JonesMatrix([[%s,%s],[%s,%s]]) at index0 & wavelength0' \
               % (self.matrix[0,0,0,0], self.matrix[0,0,0,1], self.matrix[0,0,1,0], self.matrix[0,0,1,1])

    def get_matrix(self):
        return self.matrix


class Rotater(JonesMatrix):
    def __init__(self, angle, dimension):
        angle = np.radians(angle)
        matrix = [[np.cos(angle) * np.ones(dimension), -np.sin(angle) * np.ones(dimension)],
                  [np.sin(angle) * np.ones(dimension), np.cos(angle) * np.ones(dimension)]]
        matrix = np.array(matrix).transpose((2, 3, 0, 1))
        super(Rotater, self).__init__(matrix)


class Element(JonesMatrix):
    def __init__(self, Ax, Ay, phix, phiy):
        matrix = [[Ax * np.exp(1j * phix), np.zeros(Ax.shape)], [np.zeros(Ax.shape), Ay * np.exp(1j * phiy)]]
        # [2 2 729 20]> [729 20 2 2]
        matrix = np.array(matrix).transpose((2, 3, 0, 1))
        super(Element, self).__init__(matrix)

