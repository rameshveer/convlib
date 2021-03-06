import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Model(nn.Module):

    def __init__(self, dropout_value=0.25):

        self.dropout_value = dropout_value  # dropout value

        super(CIFAR10Model, self).__init__()

        # cifar input images size = 32 x 32

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 32

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 32
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 16

        # CONVOLUTION BLOCK 2

        # DEPTHWISE SEPARABLE CONVOLUTION:

        # 1.DEPTHWISE CONVOLUTION:

        self.depthwise1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=0, groups=32, bias=False),
            # since groups is 32, 32 (3x3) kernels will convolve on 32 inp channels

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 16

        # 2.POINTWISE CONVOLUTION:

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 16

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 8

        # CONVOLUTION BLOCK 3

        # DILATED CONVOLUTION

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=4, dilation=2, bias=False),
            # since dilation is 2, 3x3 kernel can view a 5x5 area and same is stored as RF
            # in output channel

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 11
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 11

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2)  # output_size = 5

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.depthwise1(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool3(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)