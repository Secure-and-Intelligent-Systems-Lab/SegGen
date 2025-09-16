import torch
import torch.nn.init as init
from torch import nn, Tensor
from torch.nn import functional as F
from models.layers.common import DropPath


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

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

        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class MultiModalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, qk_scale=None, qkv_bias=False, attention_drop_rate=0.,
                 projection_drop_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop_rate)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(projection_drop_rate)

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        q = [self.q(x_) for x_ in x]
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]
        if self.sr_ratio > 1:
            x = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            x = [self.sr(x_) for x_ in x]
            x = [x_.reshape(B, C, -1).permute(0, 2, 1) for x_ in x]  # [B, Nr, C]
            x = [self.norm(x_) for x_ in x]
            kv = [self.kv(x_) for x_ in x]
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        else:
            kv = [self.kv(x_) for x_ in x]
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]

        k, v = zip(*[(modality[0], modality[1]) for modality in kv])

        outputs = []
        for i, q_ in enumerate(q):
            k_others = torch.cat([k[j] for j in range(len(k)) if j != i], dim=2)
            v_others = torch.cat([v[j] for j in range(len(v)) if j != i], dim=2)

            cross_attn = (q_ @ k_others.transpose(-2, -1)) * self.scale
            cross_attn = cross_attn.softmax(dim=-1)
            cross_attn = self.attention_drop(cross_attn)

            x_ = (cross_attn @ v_others).transpose(1, 2).reshape(B, N, C)
            x_ = self.projection(x_)
            x_ = self.projection_drop(x_)
            outputs.append(x_)

        return outputs


class MMTFBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.1, attention_drop_rate=0.1,
                 path_drop=0.1, activation_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, num_modalities=1, top=False):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.norm1 = nn.ModuleList([norm_layer(dim) for i in range(num_modalities)])
        self.norm_shared = norm_layer(dim)
        self.attention = MultiModalCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attention_drop_rate=attention_drop_rate, projection_drop_rate=drop, sr_ratio=sr_ratio)

        self.path_drop = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.linear_fuse = nn.Linear(dim * num_modalities, dim, bias=True)
        self.texture = TextureExtractor(dim, dim * 4, dim, top)
        self.activation = activation_layer

    def forward(self, x, H, W):
        B = x[0].shape[0]
        x = [x_.reshape(B, -1, H * W).transpose(2, 1) for x_ in x]
        y = [self.texture(self.norm_shared(x_), H, W) for i, x_ in enumerate(x)]
        f = self.attention([self.norm_shared(x_) for i, x_ in enumerate(x)], H, W)
        x = self.linear_fuse(torch.cat(x, dim=-1))
        x = self.activation(x)
        x = self.norm(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2)


class TextureExtractor(nn.Module):
    def __init__(self, c1, c2, c3, top=False):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.top = top
        if top:
            self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c3)

        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        if self.top:
            x3 = self.dwconv7(x, H, W)
            return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2, H, W)))


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class MixTransformer(nn.Module):
    def __init__(self, model_name: str = 'B0', modality: str = 'depth', root='c', start_dims=3):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        # self.model_name = 'B2'
        self.model_name = model_name
        self.start_dims = start_dims
        # self.model_name = 'B2' if modality == 'depth' else model_name
        embed_dims, depths = mit_settings[self.model_name]
        self.modality = modality
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.root = root

        # patch_embed
        self.patch_embed1 = PatchEmbed(start_dims, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Initialize with pretrained weights
        self.init_weights()

    def init_weights(self):
        print(f"Initializing weight for {self.modality}...")
        checkpoint = torch.load(f'{self.root}mit_{self.model_name.lower()}.pth',
                                map_location=torch.device('cpu'), weights_only=True)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if self.start_dims != 3:
            print("Start Dims not equal to 3, removing embedding weight from checkpoint.")
            checkpoint.pop('patch_embed1.proj.weight')
        msg = self.load_state_dict(checkpoint, strict=False)
        del checkpoint
        print(f"Weight init complete with message: {msg}")

    def forward(self, x: Tensor) -> list:
        B = x.shape[0]
        outs = []
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1_cam = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1_cam)

        # stage 2
        x, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x = blk(x, H, W)
        x2_cam = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2_cam)

        # stage 3
        x, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x = blk(x, H, W)
        x3_cam = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3_cam)

        # stage 4
        x, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x = blk(x, H, W)
        x4_cam = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4_cam)

        return outs


class FuseFormEncoder(nn.Module):
    def __init__(self, model_name: str = 'B3', modals: list = ['rgb', 'depth', 'event', 'lidar'], checkpoint_dir=None):
        super().__init__()
        self.model_name = model_name
        embed_dims, depths = mit_settings[self.model_name]
        self.modals = modals[1:] if len(modals) > 1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.checkpoint_dir = checkpoint_dir

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        if self.checkpoint_dir is not None:
            self.init_weights()

        if self.num_modals > 0:
            # Backbones and Fusion Block for extra modalities
            self.extra_mit = nn.ModuleList(
                [MixTransformer('B2', self.modals[i], root=checkpoint_dir, start_dims=3) for i in
                 range(self.num_modals)])
            self.MMDFBlock1 = MMTFBlock(dim=embed_dims[0], num_heads=1, sr_ratio=8, num_modalities=self.num_modals + 1,
                                        top=True)
            self.MMDFBlock2 = MMTFBlock(dim=embed_dims[1], num_heads=2, sr_ratio=4, num_modalities=self.num_modals + 1,
                                        top=True)
            self.MMDFBlock3 = MMTFBlock(dim=embed_dims[2], num_heads=5, sr_ratio=2, num_modalities=self.num_modals + 1,
                                        top=False)
            self.MMDFBlock4 = MMTFBlock(dim=embed_dims[3], num_heads=8, sr_ratio=1, num_modalities=self.num_modals + 1,
                                        top=False)


    def init_weights(self):
        print(f"Initializing {self.model_name} weight for RGB...")
        checkpoint = torch.load(f'{self.checkpoint_dir}mit_{self.model_name.lower()}.pth',
                                map_location=torch.device('cpu'), weights_only=True)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        msg = self.load_state_dict(checkpoint, strict=False)
        del checkpoint
        print(f"Weight init complete with message: {msg}")

    def forward(self, x: list) -> list:
        outs = []
        if self.num_modals > 0:
            x_others = x[1:]
        x = x[0]
        B = x.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1_cam = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_others[i], _, _ = self.extra_mit[i].patch_embed1(x_others[i])
                for blk in self.extra_mit[i].block1:
                    x_others[i] = blk(x_others[i], H, W)
                x_others[i] = self.extra_mit[i].norm1(x_others[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused = self.MMDFBlock1([x1_cam, *x_others], H, W)
            outs.append(x_fused)
        else:
            outs.append(x1_cam)

        # stage 2
        x, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x = blk(x, H, W)
        x2_cam = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_others[i], _, _ = self.extra_mit[i].patch_embed2(x_others[i])
                for blk in self.extra_mit[i].block2:
                    x_others[i] = blk(x_others[i], H, W)
                x_others[i] = self.extra_mit[i].norm2(x_others[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused = self.MMDFBlock2([x2_cam, *x_others], H, W)
            outs.append(x_fused)
        else:
            outs.append(x2_cam)

        # stage 3
        x, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x = blk(x, H, W)
        x3_cam = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_others[i], _, _ = self.extra_mit[i].patch_embed3(x_others[i])
                for blk in self.extra_mit[i].block3:
                    x_others[i] = blk(x_others[i], H, W)
                x_others[i] = self.extra_mit[i].norm3(x_others[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused = self.MMDFBlock3([x3_cam, *x_others], H, W)
            outs.append(x_fused)
        else:
            outs.append(x3_cam)

        # stage 4
        x, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x = blk(x, H, W)
        x4_cam = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_others[i], _, _ = self.extra_mit[i].patch_embed4(x_others[i])
                for blk in self.extra_mit[i].block4:
                    x_others[i] = blk(x_others[i], H, W)
                x_others[i] = self.extra_mit[i].norm4(x_others[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused = self.MMDFBlock4([x4_cam, *x_others], H, W)
            outs.append(x_fused)
        else:
            outs.append(x4_cam)

        return outs, H, W
