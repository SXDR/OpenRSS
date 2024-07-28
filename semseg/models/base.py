import torch
import math
from torch import nn
from semseg.models.backbones import *
from semseg.models.layers import trunc_normal_
from semseg.models.backbones.mit import MiTrgb

class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant, 4)
        # self.backbone_rgb = eval(backbone)(variant, 3)
        # self.backbone_rgb = eval(backbone + "rgb")(variant, 3)
        # self.backbone_thermal = eval(backbone)(variant, 1)
        # self.backbone_thermal = eval(backbone + "rgb")(variant, 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            # print(self.backbone)
            # print(torch.load(pretrained, map_location='cpu'))
            # k, v = self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            # print(len(k))
            # print(len(v))
