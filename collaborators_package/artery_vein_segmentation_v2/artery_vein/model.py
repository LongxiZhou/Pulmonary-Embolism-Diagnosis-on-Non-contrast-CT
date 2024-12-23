import torch.nn as nn
import torch


class UNet(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, in_channel=1, num_classes=1):
        super().__init__()

        channel = 16
        # encoding
        self.conv1 = nn.Sequential(
                  nn.Conv3d(in_channel, channel, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel))
        self.maxpool1 = nn.Conv3d(channel, channel, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Sequential(
                  nn.Conv3d(channel, channel * 2, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 2))
        self.maxpool2 = nn.Conv3d(channel * 2, channel * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Sequential(
                  nn.Conv3d(channel * 2, channel * 4, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 4),
                  nn.Conv3d(channel * 4, channel * 4, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 4))
        self.maxpool3 = nn.Conv3d(channel * 4, channel * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Sequential(
                  nn.Conv3d(channel * 4, channel * 8, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 8),
                  nn.Conv3d(channel * 8, channel * 8, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 8))
        self.maxpool4 = nn.Conv3d(channel * 8, channel * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # center
        self.center = nn.Sequential(
                  nn.Conv3d(channel * 8, channel * 16, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 16))

        # decoding
        self.decode4 = nn.Sequential(
                  nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
                  nn.Conv3d(channel * 16, channel * 8, kernel_size=(1, 1, 1)),
                  nn.PReLU(),
                  nn.Conv3d(channel * 8, channel * 8, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 8))
        self.decode3 = nn.Sequential(
                  nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
                  nn.Conv3d(channel * 16, channel * 8, kernel_size=(1, 1, 1)),
                  nn.PReLU(),
                  nn.Conv3d(channel * 8, channel * 4, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 4))
        self.decode2 = nn.Sequential(
                  nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
                  nn.Conv3d(channel * 8, channel * 4, kernel_size=(1, 1, 1)),
                  nn.PReLU(),
                  nn.Conv3d(channel * 4, channel * 2, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 2))
        self.decode1 = nn.Sequential(
                  nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
                  nn.Conv3d(channel * 4, channel * 2, kernel_size=(1, 1, 1)),
                  nn.PReLU(),
                  nn.Conv3d(channel * 2, channel * 1, kernel_size=3, padding=1, stride=1),
                  nn.PReLU(),
                  nn.BatchNorm3d(channel * 1))

        self.final = nn.Sequential(nn.Conv3d(channel * 2, num_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                             padding=(1, 1, 1)))
                                   # nn.Softmax(dim=1))

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        # print(conv2.shape)
        maxpool2 = self.maxpool2(conv2)
        # print(maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        # print(conv3.shape)
        maxpool3 = self.maxpool3(conv3)
        # print(maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        # print(conv4.shape)
        maxpool4 = self.maxpool4(conv4)
        # print(maxpool4.shape)

        # center
        center = self.center(maxpool4)
        # print(center.shape)

        # decoding
        decode4 = self.decode4(center)
        decode4 = torch.cat([decode4, conv4], 1)
        decode3 = self.decode3(decode4)
        decode3 = torch.cat([decode3, conv3], 1)
        decode2 = self.decode2(decode3)
        decode2 = torch.cat([decode2, conv2], 1)
        decode1 = self.decode1(decode2)
        decode1 = torch.cat([decode1, conv1], 1)

        final = self.final(decode1)

        return final
