import torch.nn as nn


class Net(nn.Module):
    def __init__(self, drop=0):
        self.drop = drop
        super(Net, self).__init__()
        # Input size = 32
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )  # output_size = 32  RF = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 32  RF = 5

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 32  RF = 7

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )   # output_size = 16  RF = 8

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 16  RF = 12

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 16  RF = 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 16  RF = 20

        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )   # output_size = 8  RF = 22

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 8  RF = 30

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 8  RF = 38

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(drop)
        )   # output_size = 8  RF = 46

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0)
        )

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x1 + x2)
        x4 = self.pool1(x1 + x2 + x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x4 + x5)
        x7 = self.conv5(x4 + x5 + x6)
        x8 = self.pool2(x5 + x6 + x7)
        x9 = self.conv6(x8)
        x10 = self.conv7(x8 + x9)
        x11 = self.conv8(x8 + x9 + x10)
        x12 = self.gap(x11)
        x13 = self.fc(x12)
        x13 = x13.view(-1, 10)
        return x13
