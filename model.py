# PyTorch implementation of "Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation"
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels=64):  # Adjusted channels
        super(GhostModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels // 2)
        self.sep_conv = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1, groups=out_channels // 2, bias=False)

    def forward(self, x):
        x1 = F.relu(self.batch_norm(self.conv1(x)))
        x2 = F.relu(self.sep_conv(x1))
        return torch.cat([x1, x2], dim=1)

class HalfUNet(nn.Module):
    def __init__(self, in_channels=1):
        super(HalfUNet, self).__init__()
        self.down1 = nn.Sequential(
            GhostModule(in_channels),
            GhostModule(64),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            GhostModule(64),
            GhostModule(64),
            nn.MaxPool2d(2)
        )
        self.down3 = nn.Sequential(
            GhostModule(64),
            GhostModule(64),
            nn.MaxPool2d(2)
        )
        self.down4 = nn.Sequential(
            GhostModule(64),
            GhostModule(64),
            nn.MaxPool2d(2)
        )
        self.down5 = nn.Sequential(
            GhostModule(64),
            GhostModule(64),
            nn.MaxPool2d(2)
        )
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # activation will be applied later

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        up5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        up4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True)
        up3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        up2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)

        combined = x1 + up2 + up3 + up4 + up5
        out = GhostModule(64)(GhostModule(64)(combined))
        return torch.sigmoid(self.final(out))  # Apply sigmoid activation

model = HalfUNet()
