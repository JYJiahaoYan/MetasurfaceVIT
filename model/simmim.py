
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_para == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask):
        _, _, H, W = x.shape
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape  # [B, num_patches, embed_dim]

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]  # don't need cls token
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, loss_type, is_recon):
        super().__init__()
        self.encoder = encoder
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.loss_type = loss_type
        self.is_recon = is_recon
        self.decoder = nn.Conv2d(in_channels=self.encoder.num_features, out_channels=self.in_chans, kernel_size=1)

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).contiguous()
        if self.loss_type == 0:
            loss = F.l1_loss(x, x_rec, reduction='mean')
        elif self.loss_type == 1:
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        elif self.loss_type == 2:
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            mask = 1 - mask
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        else:
            raise ValueError("Loss Type must be 0, 1, 2!")
        if self.is_recon:
            return loss, x_rec
        else:
            return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    encoder = VisionTransformerForSimMIM(
        x_size=config.DATA.SIZE_X,
        y_size=config.DATA.SIZE_Y,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_para=0,
        embed_dim=config.MODEL.VIT.EMBED_DIM,
        depth=config.MODEL.VIT.DEPTH,
        num_heads=config.MODEL.VIT.NUM_HEADS,
        mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        qkv_bias=config.MODEL.VIT.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=config.MODEL.VIT.INIT_VALUES,
        use_abs_pos_emb=config.MODEL.VIT.USE_APE,
        use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
        use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
        use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)

    model = SimMIM(encoder, config.MODEL.LOSS_TYPE, config.RECON_MODE)

    return model
