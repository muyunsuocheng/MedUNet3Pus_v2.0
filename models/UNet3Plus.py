import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, train_data=None, labels=None,
                 epochs=None, batch_size=None):
        super(UNet3Plus, self).__init__()
        self.train_data = train_data
        self.labels = labels
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.epochs = epochs
        self.batch_size = batch_size
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # Downsampling
        self.conv1 = ConvBlock(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(2)
        self.center = ConvBlock(filters[3], filters[4], is_batchnorm)

        # Upsampling and skip connections
        self.up_concat4 = UpConvBlock(filters[4], filters[3], filters[3], is_deconv)
        self.up_concat3 = UpConvBlock(filters[3], filters[2], filters[2], is_deconv)
        self.up_concat2 = UpConvBlock(filters[2], filters[1], filters[1], is_deconv)
        self.up_concat1 = UpConvBlock(filters[1], filters[0], filters[0], is_deconv)
        
        # Final convolution
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        # Center
        center = self.center(maxpool4)
        
        # Decoder with skip connections
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)
        
        return self.final(up1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, is_deconv):
        super().__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(skip_channels + out_channels, out_channels, True)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)