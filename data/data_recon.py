import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from datetime import datetime


def build_loader_recon(config, logger):
    data_recon, mask_recon = build_data(config)
    dataset = MyDataSet(data_recon, mask_recon)
    logger.info(f"Build dataset: prediction images = {len(dataset)}")

    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=config.DATA.PIN_MEMORY, drop_last=False, collate_fn=dataset.collate_fn)

    return data_loader


def build_data(config):
    if config.TREATMENT is None or config.TREATMENT == '':
        config.defrost()
        config.TREATMENT = find_latest(config.RECON_PATH)
        config.freeze()
    file = config.RECON_PATH + '/type_' + str(config.RECON_TYPE) + '_JM_' + config.TREATMENT + '.npy'
    JM_all = np.load(file).astype('float16')
    mask_file = config.RECON_PATH + '/type_' + str(config.RECON_TYPE) + '_mask_' + config.TREATMENT + '.npy'
    mask_all = np.load(mask_file).astype('float16')
    if mask_all.ndim != JM_all.ndim:
        JM_all = np.repeat(JM_all[np.newaxis, :], mask_all.shape[0], axis=0)
    mask_all = mask_all.reshape((-1, mask_all.shape[-2], mask_all.shape[-1]))
    JM_all = JM_all.reshape((-1, JM_all.shape[-2], JM_all.shape[-1]))

    JM_all = np.expand_dims(JM_all, axis=1)  # [num, 1, num_wave, 6]
    mask_all = np.expand_dims(mask_all, axis=1)
    mask_with_index = dict((k, v) for k, v in enumerate(mask_all))
    JM_with_index = dict((k, v) for k, v in enumerate(JM_all))

    return JM_with_index, mask_with_index


def find_latest(path):
    date_pattern = r'(\d{4}-\d{2}-\d{2})\.npy$'

    latest_date = None

    for filename in os.listdir(path):
        match = re.search(date_pattern, filename)
        if match:
            file_date_str = match.group(1)
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

            if latest_date is None or file_date > latest_date:
                latest_date = file_date

    return latest_date.strftime('%Y-%m-%d') if latest_date else None


class MyDataSet(Dataset):
    def __init__(self, spectra: dict, masks: dict):
        self.spectra = spectra
        self.masks = masks

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, item):
        spec = self.spectra[item]
        mask = self.masks[item]
        return torch.tensor(spec), torch.tensor(mask)

    @staticmethod
    def collate_fn(batch):
        spectra, mask = tuple(zip(*batch))
        spectra = torch.stack(spectra, dim=0)
        mask = torch.stack(mask, dim=0)
        return spectra, mask
