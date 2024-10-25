import glob
import os.path
import random
import numpy as np
from PIL import Image
from utils import srgb_to_linear


class ImageGenerator:
    def __init__(self, args):
        self.size = args.size
        self.path = args.image_path
        self.colorful = True if args.design_type == 3 else False
        self.seed = args.random_seed
        # the maximum amount of figures that will be used
        # 6 is for one wavelength channel, which needs 6 figures (for amp11, amp12, amp22, phase11, phase12, phase22)
        self.max_num = 6
        if args.design_type == 2:
            # if haven't selected wavelengths, randomly pick three wavelengths(3x6)
            self.max_num = 6 * len(args.fixed_wave) if args.fixed_wave else 18

    def load_images(self):
        if self.colorful:
            path = os.path.join(self.path, 'color')
        else:
            path = os.path.join(self.path, 'grey')
        # find all jpg images in this folder
        jpg_files = glob.glob(os.path.join(path, "*.jpg"))
        if self.max_num > len(jpg_files):
            raise ValueError("Current image folder doesn't have enough pictures to continue metasurface design!")
        selected_files = random.sample(jpg_files, self.max_num)
        images = []
        for file in selected_files:
            img = Image.open(file)
            if not self.colorful:
                img = img.convert('L')
            img = self.cut_adjust_image(img)
            img = np.asarray(img)
            images.append(img)
        norm_images = [self.normalize(item) for item in images]
        return images, norm_images

    def cut_adjust_image(self, image):
        size = image.size
        if size[0] > size[1]:
            cut = ((size[0] - size[1]) // 2, 0, (size[0] + size[1]) // 2, size[1])
        elif size[0] < size[1]:
            cut = (0, (size[1] - size[0]) // 2, size[0], (size[1] + size[0]) // 2)
        else:
            cut = (0, 0, size[0], size[0])
        # img.crop(a0, b0, a1, b1) 0:topleft 1:bottomright
        image = image.crop(cut)
        # change the square to the size that you wanted
        image = image.resize((self.size, self.size), Image.LANCZOS)
        return image

    def normalize(self, matrix):
        if matrix.max() > 1:
            matrix = matrix.astype(float) / 255
        matrix = srgb_to_linear(matrix)
        matrix = np.sqrt(matrix)
        matrix = (matrix - 0.5) / 0.5
        return matrix
