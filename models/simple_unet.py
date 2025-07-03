import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(x.size()[2]//2,x.size()[3]//2), mode='area')
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.reduce_channels = nn.Conv2d(in_channels*2 if bilinear else in_channels, in_channels // 2, kernel_size=1)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

            # After upsampling, we have in_channels channels
            # We need to reduce it to in_channels // 2 to match the skip connection
            # After concatenation, the input channels will be in_channels//2 + in_channels//2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concatenation, the input channels will be in_channels//2 + in_channels//2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if hasattr(self, 'up_conv'):
            x1 = self.up_conv(x1)
        x2 = self.reduce_channels(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.interpolate(x1, size=(x2.size()[2], x2.size()[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, features=[64, 128, 256, 512], bilinear=False):
        super(SimpleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, features[0]))
        self.down1 = (Down(features[0], features[1]))
        self.down2 = (Down(features[1], features[2]))
        self.down3 = (Down(features[2], features[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(features[3], features[3] // factor))
        self.up1 = (Up(features[3] // factor, features[2] // factor, bilinear))
        self.up2 = (Up(features[2] // factor, features[1] // factor, bilinear))
        self.up3 = (Up(features[1] // factor, features[0] // factor, bilinear))
        self.up4 = (Up(features[0] // factor, features[0] // factor, bilinear))
        self.outc = (OutConv(features[0] // factor, out_channels))

    def forward(self, x,return_features=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if return_features:
            return logits,x1,x2,x3,x4,x5
        else:
            return logits


class SimpleUNetMask(nn.Module):
    """Simple UNet specifically for mask segmentation"""
    
    def __init__(self, in_channels=3, out_channels=20, features=[64, 128, 256, 512], bilinear=False):
        super(SimpleUNetMask, self).__init__()
        self.unet = SimpleUNet(in_channels=in_channels, out_channels=out_channels, 
                              features=features, bilinear=bilinear)
    
    def forward(self, x,return_features=False):
        return self.unet(x,return_features)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)