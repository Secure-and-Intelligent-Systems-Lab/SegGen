import torch
import torch.nn as nn
import torch.nn.functional as F
from components.factory.factory import MODELS


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

@MODELS.register("UNet")
class UNet(nn.Module):
    def __init__(self, input_channels, num_classes, modals):
        super(UNet, self).__init__()

        self.input_channels = int(input_channels) * len(modals) # Placeholder, actual channels determined dynamically
        self.enc1 = DoubleConv(self.input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        inputs: list of tensors with shape [B, C, H, W] or a single tensor [B, C, H, W]
        """
        if isinstance(inputs, list):
            x = torch.cat(inputs, dim=1)  # Stack along channel dimension
        else:
            x = inputs  # Single input case

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final_conv(dec1)


# Example usage:
if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256  # Example dimensions
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)

    model = UNet(input_channels=C, num_classes=1, modals=['img', 'depth'])  # Adjust input channels accordingly
    output = model([x1, x2])  # Multi-input case
    print(output.shape)  # Should be [B, num_classes, H, W]

    model_single = UNet(input_channels=C, num_classes=1, modals=['img'])  # Single input case
    output_single = model_single(x1)  # Single input case
    print(output_single.shape)  # Should be [B, num_classes, H, W]
