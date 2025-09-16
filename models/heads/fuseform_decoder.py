# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:31:47 2023

@author: Justin
"""
import math

import torch
import torch.nn.init as init
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import MLPBlock

from models.layers.common import DropPath


# %%#########################

class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, output_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20):
        super(SegFormer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_Strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, output_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, output_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, output_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, output_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = nn.Conv2d(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(embedding_dim)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x):
        c1, c2, c3, c4 = x
        B = c1.shape[0]
        H = c4.shape[-1]
        W = c4.shape[-2]
        # c1 = c1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # c2 = c2.reshape(B, H//2, W//2, -1).permute(0, 3, 1, 2).contiguous()
        # c3 = c3.reshape(B, H//4, H//4, -1).permute(0, 3, 1, 2).contiguous()
        # c4 = c4.reshape(B, H//8, H//8, -1).permute(0, 3, 1, 2).contiguous()
        ############## MLP decoder on C1-C4 ###########

        c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        c4 = torch.nn.functional.interpolate(c4, size=c1[0].size()[1:], mode='bilinear', align_corners=False)

        c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        c3 = torch.nn.functional.interpolate(c3, size=c1[0].size()[1:], mode='bilinear', align_corners=False)

        c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        c2 = torch.nn.functional.interpolate(c2, size=c1[0].size()[1:], mode='bilinear', align_corners=False)

        c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        x = self.bn(_c)
        x = self.activate(x)
        x = self.dropout(x)

        return x


# %%
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TFAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attention_drop_rate=0., projection_drop_rate=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by NumHeads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop_rate)
        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(projection_drop_rate)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            kv = self.kv(x)
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x)
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.projection_drop(x)

        return x


class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)


class TextureExtractor(nn.Module):
    def __init__(self, c1, c2, top=False):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.Top = top
        if top:
            self.dwconv5 = CustomDWConv(c2, 5)
            self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:

        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        if self.Top:
            x2 = self.dwconv5(x, H, W)
            x3 = self.dwconv7(x, H, W)
            return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))
        return self.fc2(F.gelu(self.pwconv2(x + x1, H, W)))


class FinalpatchexpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=True)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attention_drop_rate=0.,
                 path_drop=0., activation_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, DimOut=None, top=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attention = TFAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attention_drop_rate=attention_drop_rate, projection_drop_rate=drop, sr_ratio=sr_ratio)
        self.path_drop = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, activation_layer=activation_layer, drop=drop)
        self.bn = nn.BatchNorm2d(dim)
        self.local_extractor = TextureExtractor(dim, dim, top)
        self.feature_fuse2 = nn.Linear(dim * 2, dim, bias=True)
        self.eps = 1e-8

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B = x.shape[0]
        y = self.local_extractor(self.norm1(x), H, W)
        f = self.attention(self.norm1(x), H, W)
        f = self.feature_fuse2(torch.cat([f, y], dim=-1))
        f = self.path_drop(f)
        x = x + f
        f = self.path_drop(self.mlp(self.norm2(x), H, W))
        x = x + f

        return x


class FuseBlock(nn.Module):
    def __init__(self, in_channels=128, factor=2):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.factor = factor
        self.BN = nn.ModuleList([nn.BatchNorm2d(in_channels) for i in range(2)])
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.Projection = nn.Sequential(nn.Linear(in_channels, in_channels * 2),
                                        nn.ReLU(),
                                        nn.Linear(in_channels * 2, in_channels))
        self.BN2 = nn.BatchNorm2d(in_channels)
        self.Activation = nn.ReLU()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x, x2):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        B, C, H, W = x2.shape
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + 1e-8)
        x = self.Conv1(torch.cat([x * fuse_weights[0], x2 * fuse_weights[1]], dim=1))
        x = x.flatten(2).transpose(1, 2)
        x = self.Projection(x)
        x = self.Activation(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.Activation(x)

        return x


class FuseFormDecoder(nn.Module):

    def __init__(self, image_size=256, embedding_dims=[512, 256, 128, 64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[2, 2, 2, 2], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attention_drop_rate=0., path_drop_rate=0.,
                 decoder_layers=[2, 2, 2, 2], sr_ratios=[1, 2, 4, 8], skip=False,
                 modalities=4, mmdf_type=None):

        super().__init__()
        self.modalities = modalities
        self.image_size = image_size
        norm_layer = nn.LayerNorm
        progressive_drop_rate = [x.item() for x in torch.linspace(0, path_drop_rate, sum(decoder_layers))]
        self.channel_reduction = nn.ModuleList([nn.Linear(embedding_dims[i], embedding_dims[i + 1]) for i in range(3)])

        cur = 0

        self.block1 = nn.ModuleList([DecoderBlock(
            dim=embedding_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attention_drop_rate=attention_drop_rate, path_drop=progressive_drop_rate[cur + i],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(decoder_layers[0])])

        cur += decoder_layers[0]

        self.fuse_block2 = FuseBlock(embedding_dims[1])
        self.block2 = nn.ModuleList([DecoderBlock(
            dim=embedding_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attention_drop_rate=attention_drop_rate, path_drop=progressive_drop_rate[cur + i],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], top=False)
            for i in range(decoder_layers[1])])

        cur += decoder_layers[1]

        self.fuse_block3 = FuseBlock(embedding_dims[2])
        self.block3 = nn.ModuleList([DecoderBlock(
            dim=embedding_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attention_drop_rate=attention_drop_rate, path_drop=progressive_drop_rate[cur + i],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], top=True)
            for i in range(decoder_layers[2])])

        self.fuse_block4 = FuseBlock(embedding_dims[3])
        self.block4 = nn.ModuleList([DecoderBlock(
            dim=embedding_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attention_drop_rate=attention_drop_rate, path_drop=progressive_drop_rate[cur + i],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], top=True)
            for i in range(decoder_layers[3])])

        self.final_expand = FinalpatchexpandX4((256, 256), dim=64)

    def forward(self, x):
        c4, c3, c2, c1 = x
        B = c1.shape[0]
        h1, w1 = c1.shape[2:]
        h2, w2 = c2.shape[2:]
        h3, w3 = c3.shape[2:]
        h4, w4 = c4.shape[2:]

        # stage 1
        x = c1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block1):
            x = blk(x, h1, w1)
        x = self.channel_reduction[0](x)
        x = x.reshape(B, h1, w1, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x = self.fuse_block2(x, c2)
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block2):
            x = blk(x, h2, w2)
        x = self.channel_reduction[1](x)
        x = x.reshape(B, h2, w2, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x = self.fuse_block3(x, c3)
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            x = blk(x, h3, w3)
        x = self.channel_reduction[2](x)
        x = x.reshape(B, h3, w3, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x = self.fuse_block4(x, c4)
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block4):
            x = blk(x, h4, w4)

        # x = self.bn(x)
        x = self.final_expand(x, h4, w4)
        x = x.reshape(B, h4 * 2, w4 * 2, -1).permute(0, 3, 1, 2).contiguous()

        return x
