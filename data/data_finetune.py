import numpy as np
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def build_loader_finetune(config, logger):
    para_train, JM_train, para_val, JM_val = build_dataset(config)
    dataset_train = MyDataSet(JM_train, para_train)
    dataset_val = MyDataSet(JM_val, para_val)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    if torch.cuda.device_count() != 1:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

        data_loader_train = DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
            collate_fn=dataset_train.collate_fn
        )

        data_loader_val = DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            collate_fn=dataset_val.collate_fn
        )
    else:
        data_loader_train = DataLoader(dataset_train, config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=True, collate_fn=dataset_train.collate_fn)
        data_loader_val = DataLoader(dataset_val, config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=True, collate_fn=dataset_val.collate_fn)

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(config, val_ratio=0.8):
    # for finetune data, there is only one file for JM, while JMs were separated to several files for pretrained data.
    param_path = config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_PARAM + config.DATA.SUFFIX
    JM_path = config.DATA.PATH + config.DATA.FOLDER_NAME + config.DATA.PREFIX_JM + str(0) + config.DATA.SUFFIX
    assert os.path.exists(param_path), "dataset root: {} does not exist.".format(param_path)
    assert os.path.exists(JM_path), "dataset root: {} does not exist.".format(JM_path)

    JM = np.loadtxt(JM_path, dtype='float16')
    para = np.loadtxt(param_path, dtype='float16')  # [num, 6]
    # 0-1 normalize of parameters
    para = (para - para.min(axis=0)) / (para.max(axis=0) - para.min(axis=0))

    JM = JM.reshape((-1, config.DATA.SIZE_X, config.DATA.SIZE_Y))
    JM = np.expand_dims(JM, axis=1)  # e.g. [num, 1, 20, 6]

    total_samples = len(para)
    train_samples = int(total_samples * val_ratio)  # 80% for training
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:]

    # Create train sets
    para_train = {i: para[idx] for i, idx in enumerate(train_indices)}
    JM_train = {i: JM[idx] for i, idx in enumerate(train_indices)}

    # Create validation sets
    para_val = {i: para[idx] for i, idx in enumerate(val_indices)}
    JM_val = {i: JM[idx] for i, idx in enumerate(val_indices)}

    return para_train, JM_train, para_val, JM_val


class MyDataSet(Dataset):
    def __init__(self, spectra: dict, parameters: dict):
        self.spectra = spectra
        self.parameters = parameters

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, item):
        spec = self.spectra[item]
        para = self.parameters[item]
        return torch.tensor(spec), torch.tensor(para)

    @staticmethod
    def collate_fn(batch):
        spectra, para = tuple(zip(*batch))
        spectra = torch.stack(spectra, dim=0)
        para = torch.stack(para, dim=0)
        return spectra, para


def build_loader_prediction(config, logger):
    para_random, JM_pred = build_dataset_pred(config)
    dataset_pred = MyDataSet(spectra=JM_pred, parameters=para_random)
    logger.info(f"Build dataset: reconstructed JM for prediction = {len(dataset_pred)}")

    if torch.cuda.device_count() != 1:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = DistributedSampler(
            dataset_pred, num_replicas=num_tasks, rank=global_rank
        )

        data_loader_pred = DataLoader(
            dataset_pred, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
            collate_fn=dataset_pred.collate_fn
        )

    else:
        data_loader_pred = DataLoader(dataset_pred, config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=True, collate_fn=dataset_pred.collate_fn)

    return dataset_pred, data_loader_pred


def build_dataset_pred(config):
    JM = mix_recon_real(config)
    JM = np.expand_dims(JM, axis=1)
    para = np.zeros((JM.shape[0], 6), dtype='float16')
    para_with_index = dict((k, v) for k, v in enumerate(para))
    JM_with_index = dict((k, v) for k, v in enumerate(JM))

    return para_with_index, JM_with_index


def mix_recon_real(config):
    """
    notice: this method loads designed JM and masks. For unmasked part, this method uses your designed values, while for
    masked part, it uses the reconstructed values from the pre-trained model
    """
    # load the reconstructed JM:
    recon_path = config.RECON_PATH + '/reconJMs/type_' + str(config.RECON_TYPE) + '_' + config.TREATMENT + '.txt'
    pure_recon = np.loadtxt(recon_path, dtype='float16').reshape((-1, config.DATA.SIZE_X, config.DATA.SIZE_Y))
    # load the designed JM and masks (will need to duplicate JM on a new 0-axis to match with masks)
    mask_path = config.RECON_PATH + '/type_' + str(config.RECON_TYPE) + '_mask_' + config.TREATMENT + '.npy'
    JM_path = config.RECON_PATH + '/type_' + str(config.RECON_TYPE) + '_JM_' + config.TREATMENT + '.npy'
    JM_all = np.load(JM_path).astype('float16')
    mask_all = np.load(mask_path).astype('float16')
    if mask_all.ndim != JM_all.ndim:
        JM_all = np.repeat(JM_all[np.newaxis, :], mask_all.shape[0], axis=0)
    # mask could have dimension [num_block,size,size,num_waves,6], should change to [num_block*size*size,num_waves,6]
    # or have dimension [size,size,num_waves,6] , should change to [size*size,num_waves,6]
    mask_all = mask_all.reshape((-1, mask_all.shape[-2], mask_all.shape[-1]))
    JM_all = JM_all.reshape((-1, JM_all.shape[-2], JM_all.shape[-1]))

    # comments to understand mask mechanism:::
    # let's say, we designed a single channel JM at wave0, so mask at wave0 should be 0
    # while masks of other wavelengths are 1 (codes in evaluation/metasurface_design/...)
    # so for reconstruct JM, wave0 will be covered, while other waves will be preserved.
    # for designed JM, wave0 will be preserved, but other waves will be covered.
    JM = pure_recon * mask_all + (1 - mask_all) * JM_all
    return JM


def build_loader_metalens(JM, config, logger):
    para_random, JM_pred = build_dataset_lens(JM)
    dataset_pred = MyDataSet(spectra=JM_pred, parameters=para_random)
    logger.info(f"Build dataset: designed JM for metalens param prediction = {len(dataset_pred)}")

    data_loader_pred = DataLoader(dataset_pred, config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=True, collate_fn=dataset_pred.collate_fn)

    return dataset_pred, data_loader_pred


def build_dataset_lens(JM):
    JM = JM.astype('float16')
    JM = np.expand_dims(JM, axis=1)  # [num, 1, 20, 6]
    para = np.zeros((JM.shape[0], 6), dtype='float16')
    para_with_index = dict((k, v) for k, v in enumerate(para))
    JM_with_index = dict((k, v) for k, v in enumerate(JM))

    return para_with_index, JM_with_index