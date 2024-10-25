
import random
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset


class MaskGenerator:
    """
     mask_type(copy from argsparse):
     "mask type for pretraining: 0 for randomly selecting 1-5;"
     "1 for masking n-1 wavelength channels and only preserving one wavelength's Jones matrix"
     "2 for keeping all amplitude components (all wavelengths) but only keeping one "
     "wavelength's phase components"
     "3 uses the same masking mechanism as 1 but only keeps 11 polarized component and mask"
     "12 and 22 components"
     "4 uses the same masking mechanism as 2 but only keeps 11 polarized component and mask"
     "12 and 22 components"
     "5 masks 12 and 22 polarized components for all wavelengths but keeps 11 polarized "
     "components for all wavelengths")
    """
    def __init__(self, input_x=20, input_y=6):
        self.input_x = input_x
        self.input_y = input_y

    def __call__(self, mask_type):
        random_num1 = random.randrange(self.input_x)  # to choose one wavelength
        # assign 1 if you wanna mask, 0 if you wanna keep.
        mask = np.ones((self.input_x, self.input_y), dtype='float16')

        if mask_type == 1:
            mask[random_num1, :] = 0
        elif mask_type == 2:
            mask[:, :3] = 0
            mask[random_num1, 3:] = 0
        elif mask_type == 3:
            mask[random_num1, [0, 3]] = 0
        elif mask_type == 4:
            mask[:, 0] = 0
            mask[random_num1, 3] = 0
        elif mask_type == 5:
            mask[:, [0, 3]] = 0
        else:
            raise ValueError('mask_type should be int: 0 1 2 3 4 5')

        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        return mask


class DataTransform:
    def __init__(self, config):
        self.mask_generator = MaskGenerator(input_x=config.DATA.SIZE_X, input_y=config.DATA.SIZE_Y)
        self.mask_type = config.DATA.MASK_TYPE

    def __call__(self, img):
        img = torch.tensor(img)
        self.mask_type = random.randrange(1, 6) if self.mask_type == 0 else self.mask_type
        mask = self.mask_generator(self.mask_type)
        return img, mask


class MyDataSet(Dataset):
    def __init__(self, spectra: dict, transform=None):
        self.spectra = spectra
        self.transform = transform

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, item):
        spec = self.spectra[item]
        if self.transform is not None:
            spec, mask = self.transform(spec)
        else:
            raise ValueError('Mask has not been generated!')
        return spec, mask

    @staticmethod
    def collate_fn(batch):
        spectra, mask = tuple(zip(*batch))
        spectra = torch.stack(spectra, dim=0)
        mask = torch.stack(mask, dim=0)
        return spectra, mask


def read_split_data(config):
    param_path = config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_PARAM + config.DATA.SUFFIX
    JM_paths = [config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_JM + str(i) + config.DATA.SUFFIX
                for i in range(config.DATA.DIVIDE_NUM)]
    assert os.path.exists(param_path), "dataset root: {} does not exist.".format(param_path)
    for JM_path in JM_paths:
        assert os.path.exists(JM_path), "dataset root: {} does not exist.".format(JM_path)

    # re-combine separated data
    para = np.loadtxt(param_path, dtype='float16')  # [total_num, 6]
    total = para.shape[0]
    batch = total // config.DATA.DIVIDE_NUM
    lenx, leny = config.DATA.SIZE_X, config.DATA.SIZE_Y
    JM = np.zeros((total, lenx * leny), dtype='float16')
    for num in range(config.DATA.DIVIDE_NUM):
        if num == config.DATA.DIVIDE_NUM-1:
            JM[num*batch:, :] = np.loadtxt(JM_paths[num], dtype='float16')
        else:
            JM[num*batch:(num+1)*batch, :] = np.loadtxt(JM_paths[num], dtype='float16')

    JM = JM.reshape((total, lenx, leny))
    JM = np.expand_dims(JM, axis=1)  # e.g. [21526641, 1, 20, 6]
    para_with_index = dict((k, v) for k, v in enumerate(para))
    JM_with_index = dict((k, v) for k, v in enumerate(JM))
    return para_with_index, JM_with_index


def build_loader_simmim(config, logger):
    transform = DataTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')
    check_all_data(config)  # only check if corresponding folders exist
    _, train_JM = read_split_data(config)
    dataset = MyDataSet(spectra=train_JM, transform=transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    if torch.cuda.device_count() != 1:
        # randomly distribute 4 parts into 4 ranks
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=dataset.collate_fn)
    else:
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=dataset.collate_fn)
    
    return dataloader


def check_all_data(config):
    num_set = set()
    for i in range(config.DATA.DATA_SIZE):
        path = config.DATA.PATH + f'training_data_{i+config.DATA.DATA_START}/params_from_preprocess.txt'
        if not os.path.exists(path):
            raise ValueError(f'Path: {path} doesnt exist, so DATA_SIZE & DATA_START setting is unmatched with '
                             f'actual number of groups of data')
        # check if the number of wavelength points is consistent among data in different folders
        with open(path, 'r') as file:
            params_list = file.read().split()
        index = params_list.index("DATA.SIZE_X") + 1
        size_x = int(params_list[index])
        num_set.add(size_x)
        if len(num_set) > 1:
            raise ValueError('data in different folders have different wavelength points')
