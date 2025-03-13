import torch 
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        downsample_1 , padding_1 = self.down_convolution_1(x)
        downsample_2 , padding_2 = self.down_convolution_2(padding_1)
        downsample_3 , padding_3 = self.down_convolution_3(padding_2)
        downsample_4 , padding_4 = self.down_convolution_4(padding_3)

        bottle_neck = self.bottle_neck(padding_4)

        upsample_1 = self.up_convolution_1(bottle_neck, downsample_4)
        upsample_2 = self.up_convolution_2(upsample_1, downsample_3)
        upsample_3 = self.up_convolution_3(upsample_2, downsample_2)
        upsample_4 = self.up_convolution_4(upsample_3, downsample_1)

        return self.out(upsample_4)

if __name__ == "__main__":
    double_conv = DoubleConv(256,256)
    print(double_conv)

    input = torch.randn(1, 3, 512, 512)
    model = UNet(3, 10)
    output = model(input)
    print(output.size())

