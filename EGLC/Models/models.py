from functools import partial
import numpy as np
from Models.utils import Bottleneck
import torch
from torch import nn
from Models.resnet import ResNet
from Models import pvt_v2
from timm.models.vision_transformer import _cfg


class IBC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.mid_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h1 = self.in_layers(x)
        h = self.mid_layers(h1)
        h = self.out_layers(h + h1)
        return h + self.skip(x)



class TB(nn.Module):
    def __init__(self, imgsize):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 3, 9, 2],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("D:/pretrainedmodel/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint, strict=False)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    IBC([64, 128, 320, 512][i], 64), IBC(64, 64), nn.Upsample(size=int(imgsize/4))
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(IBC(128, 64), IBC(64, 64), IBC(64, 64)))


    def get_pyramid(self, x, x_RES):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                if i == 0:
                    x0 = None
                else:
                    x0 = x_RES[int((i / 3) - 1)]
                x, H, W, x1 = module(x, x0)
            elif i in [1, 4, 7, 10]:
                exp_lay = 1
                for sub_module in module:
                    x, x1 = sub_module(x, H, W, exp_lay, x1)
                    exp_lay += 1
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x, x_RES):
        pyramid = self.get_pyramid(x, x_RES)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l1 = self.SFA[i][0](l)
            l2 = self.SFA[i][1](l1)
            l = l2.mul(-1 * (torch.sigmoid(l1)) + 1)
            l = self.SFA[i][2](l)
            l_i = l

        return l, pyramid


class EGIC(nn.Module):
    def __init__(self, size=352):

        super().__init__()
        self.res = ResNet(Bottleneck, [3, 4, 6, 3])
        self.TB = TB(size)


        self.PH = nn.Sequential(
            IBC(64, 64), IBC(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        features = self.res(x)
        x1, pyramid = self.TB(x, features)
        x1 = self.up_tosize(x1)
        out = self.PH(x1)

        return out, pyramid

