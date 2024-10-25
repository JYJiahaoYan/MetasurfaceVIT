import numpy as np


class JonesVector:
    eps = 1e-15

    def __init__(self, polarization, normalize=True, normal_form=True):
        """This represents a Jones vector and is one representation of the polarisation of light.

        :param polarization: An two element iterable with complex numbers representing the value and phase
                             of the Ex and Ey light component
        """
        if hasattr(polarization, '__iter__'):
            if len(polarization[0,0,:]) != 2:
                raise ValueError('Length of vector/list must be excactly 2')
            else:
                self.polarization_vector = np.matrix(polarization).astype(complex)
        else:
            raise ValueError('Parameter must be a numpy.array')

        if normalize:
            self._normalize()
        if normal_form:
            self._make_normal_form()

        for idx, num in enumerate([self.Ex, self.Ey]):
            new_real = num.real
            new_imag = num.imag
            if abs(new_real) < JonesVector.eps:
                new_real = 0.0
            if abs(new_imag) < JonesVector.eps:
                new_imag = 0.0
            self[idx] = new_real+1j*new_imag

    def __repr__(self):
        return 'JonesVector([%s, %s]) at index0 & wavelength0' \
               % (self.polarization_vector[0,0,0], self.polarization_vector[0,0,1])

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('Needs to be integer')
        elif item not in [0, 1]:
            raise IndexError('Index can be either 0 or 1')
        else:
            return self.polarization_vector[0, 0, item]

    def __setitem__(self, index, wavelength, key, value):
        if not isinstance(key, int):
            raise TypeError('Needs to be integer')
        elif key not in [0, 1]:
            raise IndexError('Index can be either 0 or 1')
        elif not isinstance(value, (int, float, complex)):
            raise ValueError('Value needs to be numeric')
        else:
            self.polarization_vector[index, wavelength, key] = value

    def _normalize(self, index, wavelength):
        self.intensity = np.sum(np.square(np.abs(self.polarization_vector[index, wavelength, :])))
        self.polarization_vector[index, wavelength, :] /= np.sqrt(self.intensity)

    def _make_normal_form(self):
        E_x_abs = np.abs(self.Ex)
        E_y_abs = np.abs(self.Ey)
        phi_x = np.angle(self.Ex)  # math.atan(b/a)
        phi_y = np.angle(self.Ey)
        phi_y_rotated = phi_y - phi_x
        E_x_new = E_x_abs * np.exp(1j*0)
        E_y_new = E_y_abs * np.exp(1j*phi_y_rotated)
        self[0] = E_x_new
        self[1] = E_y_new

    @property
    def Ex(self):
        """Property which returns the x component of the electric field

        :return: The x component of the electric field
        :rtype: complex
        """
        return self[0]

    @property
    def Ey(self):
        """Property which returns the y component of the electric field

        :return: The y component of the electric field
        :rtype: complex
        """
        return self[1]