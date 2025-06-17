"""
Copyright (c) Meta Platforms, Inc. and affiliates.

UNet implementation for the UnSAMFlow project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block with optional batch normalization"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if use_batchnorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_batchnorm=use_batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=64, bilinear=True, use_batchnorm=True):
        """
        UNet architecture for image segmentation or dense prediction tasks
        
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output channels
            n_filters (int): Number of filters in the first layer
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions
            use_batchnorm (bool): Whether to use batch normalization
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv(n_channels, n_filters, use_batchnorm=use_batchnorm)
        self.down1 = Down(n_filters, n_filters * 2, use_batchnorm=use_batchnorm)
        self.down2 = Down(n_filters * 2, n_filters * 4, use_batchnorm=use_batchnorm)
        self.down3 = Down(n_filters * 4, n_filters * 8, use_batchnorm=use_batchnorm)
        self.down4 = Down(n_filters * 8, n_filters * 16 // factor, use_batchnorm=use_batchnorm)
        
        # Decoder path
        self.up1 = Up(n_filters * 16, n_filters * 8 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up2 = Up(n_filters * 8, n_filters * 4 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up3 = Up(n_filters * 4, n_filters * 2 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up4 = Up(n_filters * 2, n_filters, bilinear, use_batchnorm=use_batchnorm)
        
        self.outc = OutConv(n_filters, n_classes)

    def forward(self, x):
        # Encoder path - with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path - with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return logits
    
    def init_weights(self):
        """Initialize the weights for training from scratch"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MaskUNet(nn.Module):
    """UNet with a mask for the output"""
    def __init__(self, n_channels=3, n_classes=20, n_filters=64, bilinear=True, use_batchnorm=True):
        super(MaskUNet, self).__init__()
        self.unet = UNet(n_channels, n_classes, n_filters, bilinear, use_batchnorm)
        self.mask_conv = nn.Conv2d(n_classes, 1, kernel_size=1)

    def forward(self, x):
        logits = self.unet(x)
        mask = self.mask_conv(logits)
        return logits, mask