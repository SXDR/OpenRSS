import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
import torch.nn as nn


class TransBottleneck_0630(nn.Module):

    def __init__(self, inplanes, planes, stride=1, up=0):
        super(TransBottleneck_0630, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.conv3 = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.up = up

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        if self.up:
            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)

        return out


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class TransBottleneck_one(nn.Module):

    def __init__(self, inplanes, planes, stride=1, reduction=4):
        super(TransBottleneck_one, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.conv3 = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # residual = self.conv3(x)
        # residual = self.bn3(residual)
        #
        # out += residual
        # out = self.relu(out)

        return out


class TransBottleneck_one_0629(nn.Module):

    def __init__(self, inplanes, planes, stride=1, reduction=4):
        super(TransBottleneck_one_0629, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        out += residual
        out = self.relu(out)

        return out


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone_rgb.channels,
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)
        # self.decode_4 = nn.Conv2d(32, num_classes, 1)

        # self.decode_transblock1 = TransBottleneck_0630(256, 160, 2, 1)
        # self.decode_transblock2 = TransBottleneck_0630(160, 64, 2, 1)
        # self.decode_transblock3 = TransBottleneck_0630(64, 32, 2, 1)
        # self.decode_transblock4 = TransBottleneck_0630(32, 16, 2, 1)

        self.decode_transblock1 = TransBottleneck_0630(256, 160, 2, 1)
        self.decode_transblock2 = TransBottleneck_0630(320, 64, 2, 1)
        self.decode_transblock3 = TransBottleneck_0630(128, 32, 2, 1)
        self.decode_transblock4 = TransBottleneck_0630(64, 16, 2, 1)

        # self.nbt1 = non_bottleneck_1d(160, 0, 1)
        # self.nbt2 = non_bottleneck_1d(64, 0, 1)
        # self.nbt3 = non_bottleneck_1d(32, 0, 1)
        # self.nbt4 = non_bottleneck_1d(16, 0, 1)
        #
        # self.up1 = UpsamplerBlock(256, 160)
        # self.up2 = UpsamplerBlock(160, 64)
        # self.up3 = UpsamplerBlock(64, 32)
        # self.up4 = UpsamplerBlock(32, 16)

        self.nbt1 = non_bottleneck_1d(320, 0, 1)
        self.nbt2 = non_bottleneck_1d(128, 0, 1)
        self.nbt3 = non_bottleneck_1d(64, 0, 1)
        self.nbt4 = non_bottleneck_1d(16, 0, 1)

        self.up1 = UpsamplerBlock(256, 160)
        self.up2 = UpsamplerBlock(320, 64)
        self.up3 = UpsamplerBlock(128, 32)
        self.up4 = UpsamplerBlock(64, 16)

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        # x = x[:, :3]
        # x = x[:, 3:]
        rgb = x[:, :3]
        thermal = x[:, 3:]

        # y_thermal_1, y_thermal_2, y_thermal_3, y_thermal_4 = self.backbone_thermal(thermal)
        # y_rgb = self.backbone_rgb(rgb, y_thermal_1, y_thermal_2, y_thermal_3, y_thermal_4)
        y_rgb_1, y_rgb_2, y_rgb_3, y_rgb_4 = self.backbone_rgb(rgb)
        # print(y_rgb_1)
        # print(y_rgb_2)
        # print(y_rgb_3)
        # print(y_rgb_4)
        # y_rgb_1 = y_rgb_1 * qz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # y_rgb_2 = y_rgb_2 * qz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # y_rgb_3 = y_rgb_3 * qz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # y_rgb_4 = y_rgb_4 * qz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print(y_rgb_1)
        # print(y_rgb_2)
        # print(y_rgb_3)
        # print(y_rgb_4)
        y_thermal = self.backbone_thermal(thermal, y_rgb_1, y_rgb_2, y_rgb_3, y_rgb_4)
        # y = []
        # for i in range(len(y_rgb)):
        #     y.append(y_rgb[i] + y_thermal[i])
        # y = tuple(y)

        y = y_thermal
        # 用RTNet_decode
        # seg1 = self.decode_transblock1(y[3])
        # seg1 = seg1 + y[2]
        # seg1 = torch.cat((seg1, y[2]), dim=1)

        # seg2 = self.decode_transblock2(seg1)
        # seg2 = seg2 + y[1]
        # seg2 = torch.cat((seg2, y[1]), dim=1)

        # seg3 = self.decode_transblock3(seg2)
        # seg3 = seg3 + y[0]
        # seg3 = torch.cat((seg3, y[0]), dim=1)

        # seg4 = self.decode_transblock4(seg3)

        # 用nbt_decode
        # seg1 = self.up1(y[3])
        # # seg1 = seg1 + y[2]
        # seg1 = torch.cat((seg1, y[2]), dim=1)
        # seg1 = self.nbt1(seg1)
        #
        # seg2 = self.up2(seg1)
        # # seg2 = seg2 + y[1]
        # seg2 = torch.cat((seg2, y[1]), dim=1)
        # seg2 = self.nbt2(seg2)
        #
        # seg3 = self.up3(seg2)
        # # seg3 = seg3 + y[0]
        # seg3 = torch.cat((seg3, y[0]), dim=1)
        # seg3 = self.nbt3(seg3)
        #
        # seg4 = self.up4(seg3)
        # seg4 = self.nbt4(seg4)

        # 公共部分映射到原分辨率
        # output = self.output_conv(seg4)
        # y = output
        y = self.decode_head(y)  # 4x reduction in image size
        # print(y.shape)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y


class segori(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels,
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        # x = x[:, :3]
        # x = x[:, 3:]
        y = self.backbone(x)
        y = self.decode_head(y)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y


if __name__ == '__main__':
    import torch
    from thop import profile

    model = segori('MiT-B0', 9)
    # print(model.backbone)
    # print(model.decode_head)
    # print(model)
    # model.load_state_dict(torch.load('../../mit_b0.pth', map_location='cpu'),strict=False)
    x = torch.zeros(1, 4, 480, 640)
    y = model(x)
    # y = model(x, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    print(y.shape)

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
