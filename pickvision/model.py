"""PickNet: RGB-D semantic segmentation + suction affordance on a pretrained MobileNetV3 LR-ASPP."""
import torch
import torch.nn as nn
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

from pickvision.preprocess import CAM_HEIGHT, HEIGHT_RANGE, height_channel  # noqa: F401  (re-exported)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PickNet(nn.Module):
    """Input  [N, 4, H, W]: RGB in [0, 1] + height channel in [0, 1] (all zeros = RGB-only mode).
    Output [N, 4, H, W]: P(tote), P(item), P(boundary), P(suction succeeds)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = lraspp_mobilenet_v3_large(weights=w)
        old = net.backbone["0"][0]
        conv = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            conv.weight[:, :3] = old.weight
            conv.weight[:, 3:] = old.weight.mean(1, keepdim=True)
        net.backbone["0"][0] = conv
        net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, 4, 1)
        net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, 4, 1)
        self.net = net

    def logits(self, x):
        rgb = (x[:, :3] - MEAN.to(x)) / STD.to(x)
        hgt = (x[:, 3:] - 0.1) / 0.2
        return self.net(torch.cat([rgb, hgt], 1))["out"]

    def forward(self, x):
        z = self.logits(x)
        return torch.cat([torch.softmax(z[:, :3], 1), torch.sigmoid(z[:, 3:])], 1)

    def freeze_stem(self, blocks=4):
        for i in range(blocks):
            for p in self.net.backbone[str(i)].parameters():
                p.requires_grad = False
