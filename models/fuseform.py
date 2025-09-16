# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:13:32 2023

@author: Justin
"""

import torch
from torch import nn

from components.factory.factory import MODELS
from .backbones.fuseform_backbone import FuseFormEncoder
from .heads.fuseform_decoder import SegFormer, FuseFormDecoder


@MODELS.register("FuseForm")
class FuseForm(nn.Module):
    def __init__(
            self,
            backbone: str = 'FuseForm-B2-FuseForm',
            classes: int = 2,
            modalities: list = ['rgb'],
            root: str = None,
    ):
        super().__init__()
        self.backbone, self.size, self.decoder = str.split(backbone, '-')
        self.Modalities = modalities

        if backbone == 'B0':
            decoder_layers = [1, 1, 1, 1]
            embedding_dims = [32, 64, 128, 256]
        else:
            decoder_layers = [2, 2, 2, 2]
            embedding_dims = [64, 128, 320, 512]

        # Transformer

        self.transformer = FuseFormEncoder(self.size, checkpoint_dir=root, modals=modalities)

        if self.decoder == 'Segformer':
            self.Segmenter = SegFormer(feature_strides=[4, 8, 16, 32], in_channels=embedding_dims,
                                       embedding_dim=768)
            self.LinearPrediction = nn.Conv2d(768, classes, kernel_size=1, stride=1)
        elif self.decoder == 'FuseForm':
            self.Segmenter = FuseFormDecoder(embedding_dims=embedding_dims[::-1],
                                             num_heads=[8, 5, 2, 1], mlp_ratios=[2, 2, 2, 2], qkv_bias=False,
                                             drop_rate=0.1,
                                             attention_drop_rate=0.1, path_drop_rate=0.1, decoder_layers=decoder_layers,
                                             sr_ratios=[1, 2, 4, 8])
            self.LinearPrediction2 = nn.Conv2d(256, classes, kernel_size=1, stride=1)

    def forward(self, x):
        img_size = x[0].shape[-2:]
        x, H, W = self.transformer(x)  # b, gh*gw+1, d
        x = self.Segmenter(x)
        x = self.LinearPrediction2(x)
        x = torch.nn.functional.interpolate(x, size=(img_size[0], img_size[1]),
                                            mode='bilinear', align_corners=False)
        return x
