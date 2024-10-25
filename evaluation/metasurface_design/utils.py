import os
import numpy as np
import matplotlib.pyplot as plt
import random


def GSCalculation(meta_intensity, im, tolerance, visualize: bool=False, bias: float=0.0, phase_level: int=360):
    holo_intensity = im / 255 if im.max() > 1 else im
    meta_intensity = meta_intensity / 255 if meta_intensity.max() > 1 else meta_intensity
    # notice: change former black part to white part. that is, let internal pattern (rather than periphery)
    # becomes bright, which may be easier to achieve for holography
    holo_intensity = 1 - holo_intensity
    holo_intensity = srgb_to_linear(holo_intensity)
    meta_intensity = srgb_to_linear(meta_intensity)
    initial_phase = np.random.uniform(-np.pi, np.pi, (im.shape[0], im.shape[1]))
    Minimum_Phase, Error = GSMain(meta_intensity, holo_intensity, initial_phase, tolerance, phase_level)
    Minimum_Wave = np.sqrt(meta_intensity) * np.exp(1j * Minimum_Phase)
    F = np.fft.fft2(Minimum_Wave, norm="ortho")
    F_ = np.fft.fftshift(F)
    Calculated_Intensity = abs(F_) ** 2

    Error = sum(sum((np.sqrt(holo_intensity) - abs(F_)) ** 2))
    print("Final Error (Discretization) : ", Error)

    if visualize:
        x = np.arange(int(-im.shape[1] / 2), int(im.shape[1] / 2))
        y = np.arange(int(-im.shape[0] / 2), int(im.shape[0] / 2))
        plt.title('Printing Intensity')
        plt.pcolormesh(x, y, meta_intensity)
        plt.set_cmap('gray')
        plt.xlim(int(-im.shape[1] / 2), int(im.shape[1] / 2) - 1)
        plt.ylim(int(-im.shape[0] / 2), int(im.shape[0] / 2) - 1)
        plt.colorbar()
        # plt.clim(0, 1)
        plt.show()

        x = np.arange(int(-im.shape[1] / 2), int(im.shape[1] / 2))
        y = np.arange(int(-im.shape[0] / 2), int(im.shape[0] / 2))
        plt.title('Hologram Intensity')
        plt.pcolormesh(x, y, Calculated_Intensity)
        plt.set_cmap('gray')
        plt.xlim(int(-im.shape[1] / 2), int(im.shape[1] / 2) - 1)
        plt.ylim(int(-im.shape[0] / 2), int(im.shape[0] / 2) - 1)
        plt.colorbar()
        # plt.clim(0, 1)
        plt.show()

        x = np.arange(int(-im.shape[1] / 2), int(im.shape[1] / 2))
        y = np.arange(int(-im.shape[0] / 2), int(im.shape[0] / 2))
        plt.title('Designed Phase')
        plt.pcolormesh(x, y, Minimum_Phase)
        plt.set_cmap('gray')
        plt.xlim(int(-im.shape[1] / 2), int(im.shape[1] / 2) - 1)
        plt.ylim(int(-im.shape[0] / 2), int(im.shape[0] / 2) - 1)
        plt.colorbar()
        plt.clim(-3.2, 3.2)
        plt.show()

    return np.angle(np.exp(1j * (Minimum_Phase + bias)))


def GSMain(Source, Target, initial_phase, tolerance, phase_level):
    Target_Amplitude = np.sqrt(Target)
    Source_Amplitude = np.sqrt(Source)
    A = np.exp(1j * initial_phase)

    previous_error = 0
    while True:
        B = Source_Amplitude * np.exp(1j * np.angle(A))

        C = np.fft.fft2(B, norm="ortho")
        C_ = np.fft.fftshift(C)
        error_value = sum(sum((Target_Amplitude - abs(C_)) ** 2))
        print("Intensity Error : ", error_value)
        if abs(error_value - previous_error) < 10 ** (-1 * tolerance):
            break
        D = Target_Amplitude * np.exp(1j * np.angle(C_))
        D_ = np.fft.fftshift(D)
        A = np.fft.ifft2(D_, norm="ortho")
        previous_error = error_value
    return np.around(np.angle(A) / np.pi * (phase_level / 2)) / (phase_level / 2) * np.pi, error_value


def srgb_to_linear(c_srgb):
    # sRGB and linear RGB issues
    # sRGB is optimized for the human eye, so to obtain the true spectrum, it must be converted to linear RGB
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear


def LensPhase(shapex, shapey, unitx, unity, focus_length, bias_x, bias_y, wavelength, flag):
    """
    On the analytical phase calculation for metalens:
    Still using rectangular construction, then filtering out the circular contour.
    Because the method of obtaining a line first and then rotating an arc cannot handle
    the issue of different numbers of elements at different radii
    No need to divide into multiple dimer units responsible for different wavelengths
    Research on achromatic metalens has already proven that corresponding structures can be found where Δφ/Δλ is very flat
    According to other literatures, the focus length of metalens in the visible light range is basically between 50 - 300 um
    """
    x = np.arange(-shapex / 2, shapex / 2) * unitx  # e.g. -256 -- 255
    y = np.arange(-shapey / 2, shapey / 2) * unity
    X, Y = np.meshgrid(x, y, indexing='ij')
    phase_xy = 2 * np.pi / wavelength * (focus_length - np.sqrt(focus_length ** 2 + (X - bias_x) ** 2 + (Y - bias_y) ** 2))
    clip_phase = np.angle(np.exp(1j * phase_xy))
    if flag:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(phase_xy, cmap='viridis', extent=[x.min(), x.max(), y.max(), y.min()])
        plt.colorbar()
        plt.title('phase_xy before clip')
        plt.subplot(1, 2, 2)
        plt.imshow(clip_phase, cmap='viridis', extent=[x.min(), x.max(), y.max(), y.min()])
        plt.colorbar()
        plt.title('phase_xy after clip')
        plt.tight_layout()
        plt.show()
    return clip_phase





