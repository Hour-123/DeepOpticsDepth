"""
U-Net architecture for depth estimation - PyTorch version.

This is a PyTorch implementation of U-Net for estimating
depth maps from optically coded images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and optional ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=kernel_size // 2, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for depth estimation.

    This version uses F.interpolate for upsampling to handle arbitrary input sizes.
    """

    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        # Block 1
        self.down1_1 = ConvBlock(3, 32)
        self.down1_2 = ConvBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.down2_1 = ConvBlock(32, 64)
        self.down2_2 = ConvBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.down3_1 = ConvBlock(64, 128)
        self.down3_2 = ConvBlock(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.down4_1 = ConvBlock(128, 256)
        self.down4_2 = ConvBlock(256, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5 (bottleneck)
        self.down5_1 = ConvBlock(256, 512)
        self.down5_2 = ConvBlock(512, 512)

        # Decoder - use Conv2d instead of ConvTranspose2d for flexibility
        # Block 4
        self.up4_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up4_1 = ConvBlock(512, 256)
        self.up4_2 = ConvBlock(256, 256)

        # Block 3
        self.up3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up3_1 = ConvBlock(256, 128)
        self.up3_2 = ConvBlock(128, 128)

        # Block 2
        self.up2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up2_1 = ConvBlock(128, 64)
        self.up2_2 = ConvBlock(64, 64)

        # Block 1
        self.up1_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.up1_1 = ConvBlock(64, 32)
        self.up1_2 = ConvBlock(32, 32)

        # Output layer
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 3, H, W]

        Returns:
            Depth map [batch, 1, H, W] with values in [0, 1]
        """
        # Encoder
        d1_1 = self.down1_1(x)
        d1_2 = self.down1_2(d1_1)
        d2_0 = self.pool1(d1_2)

        d2_1 = self.down2_1(d2_0)
        d2_2 = self.down2_2(d2_1)
        d3_0 = self.pool2(d2_2)

        d3_1 = self.down3_1(d3_0)
        d3_2 = self.down3_2(d3_1)
        d4_0 = self.pool3(d3_2)

        d4_1 = self.down4_1(d4_0)
        d4_2 = self.down4_2(d4_1)
        d5_0 = self.pool4(d4_2)

        d5_1 = self.down5_1(d5_0)
        d5_2 = self.down5_2(d5_1)

        # Decoder with skip connections using interpolate for upsampling
        u4_up = F.interpolate(d5_2, size=d4_2.shape[2:], mode='bilinear', align_corners=False)
        u4_up = self.up4_conv(u4_up)
        u4_0 = torch.cat([u4_up, d4_2], dim=1)
        u4_1 = self.up4_1(u4_0)
        u4_2 = self.up4_2(u4_1)

        u3_up = F.interpolate(u4_2, size=d3_2.shape[2:], mode='bilinear', align_corners=False)
        u3_up = self.up3_conv(u3_up)
        u3_0 = torch.cat([u3_up, d3_2], dim=1)
        u3_1 = self.up3_1(u3_0)
        u3_2 = self.up3_2(u3_1)

        u2_up = F.interpolate(u3_2, size=d2_2.shape[2:], mode='bilinear', align_corners=False)
        u2_up = self.up2_conv(u2_up)
        u2_0 = torch.cat([u2_up, d2_2], dim=1)
        u2_1 = self.up2_1(u2_0)
        u2_2 = self.up2_2(u2_1)

        u1_up = F.interpolate(u2_2, size=d1_2.shape[2:], mode='bilinear', align_corners=False)
        u1_up = self.up1_conv(u1_up)
        u1_0 = torch.cat([u1_up, d1_2], dim=1)
        u1_1 = self.up1_1(u1_0)
        u1_2 = self.up1_2(u1_1)

        # Output layer
        output = torch.sigmoid(self.out_conv(u1_2))

        return output


# Convenience function for creating the model
def create_unet():
    """Create a UNet model."""
    return UNet()
