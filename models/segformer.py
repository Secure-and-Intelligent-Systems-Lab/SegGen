import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.nn.init as init
from models.layers.common import DropPath
from .heads.segformer import SegFormerHead
from components.factory.factory import MODELS

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))

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

@MODELS.register("MixTransformer")
class MixTransformer(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes = 7, modality: str = 'depth', root=None, start_dims = 3):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        # self.model_name = 'B2'
        self.model_name = model_name
        self.start_dims = start_dims
        # self.model_name = 'B2' if modality == 'depth' else model_name
        embed_dims, depths = mit_settings[self.model_name]
        self.modality = modality[0]
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

        self.decoder = SegFormerHead(embed_dims, num_classes=num_classes)

        # Initialize with pretrained weights
        if root is not None:
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
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # Stack along channel dimension
        else:
            x = x  # Single input case

        x = x
        img_size = x[0].shape[-2:]
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

        #final = decoder
        final = self.decoder(outs)
        final = F.interpolate(final, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)

        return final
