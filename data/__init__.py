from .data_simmim import build_loader_simmim
from .data_finetune import build_loader_finetune, build_loader_prediction
from .data_recon import build_loader_recon


def build_loader(config, logger, type):
    if type == 'pre_trained':
        return build_loader_simmim(config, logger)
    elif type == 'finetune':
        return build_loader_finetune(config, logger)
    elif type == 'reconstruct':
        return build_loader_recon(config, logger)
    elif type == 'predict':
        return build_loader_prediction(config, logger)
    else:
        raise ValueError("Invalid type (should be 'pre_trained', 'finetune', 'reconstruct', or 'predict')!")
