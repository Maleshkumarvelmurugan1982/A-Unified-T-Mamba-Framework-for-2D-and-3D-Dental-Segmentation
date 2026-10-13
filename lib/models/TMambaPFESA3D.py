"""T-Mamba 3D + PFESA on the three skip connections (parameter-free; same weights/keys as TMamba3D). Select: --model TMambaPFESA3D"""
import torch
import torch.nn as nn
from .TMamba3D import TMamba3D
from .pfesa import PFESA


class TMambaPFESA3D(TMamba3D):
    def __init__(self, in_channels: int = 1, classes: int = 1, input_size: tuple = (160, 160, 96),
                 high_freq: float = 0.9, low_freq: float = 0.1):
        super().__init__(in_channels, classes, input_size, high_freq, low_freq)
        self.model_name = "TMambaPFESA3D"
        self.pfesa = nn.ModuleList([PFESA(), PFESA(), PFESA()])   # no parameters -> adds nothing to state_dict

    def forward(self, x):
        x, skip_1 = self.dfs_blocks[0](x, self.learnable_positional_embed[0], 0, self.SpectralGatingBlocks[0], self.GateModules[0])
        x, skip_2 = self.dfs_blocks[1](x, self.learnable_positional_embed[1], 1, self.SpectralGatingBlocks[1], self.GateModules[1])
        _, skip_3 = self.dfs_blocks[2](x, self.learnable_positional_embed[2], 2, self.SpectralGatingBlocks[2], self.GateModules[2])

        skip_1 = self.pfesa[0](skip_1)          # 12 ch, 1/2 resolution
        skip_2 = self.pfesa[1](skip_2)          # 24 ch, 1/4 resolution
        skip_3 = self.pfesa[2](skip_3)          # 24 ch, 1/8 resolution

        skip_2 = self.upsample_1(skip_2)
        skip_3 = self.upsample_2(skip_3)
        out = self.out_conv(torch.cat([skip_1, skip_2, skip_3], 1))
        out = self.upsample_out(out)
        return out

