from .vision_transformer import build_vit
from .simmim import build_simmim


# model solely based on VIT
def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(config)
    else:
        model = build_vit(config)

    return model
