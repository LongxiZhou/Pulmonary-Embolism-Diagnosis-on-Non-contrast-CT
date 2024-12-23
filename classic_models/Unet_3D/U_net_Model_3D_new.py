from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial


class UNet3D(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=64, leaky=0.1, batch_norm=True):
        super(UNet3D, self).__init__()

        cnn_block = partial(block_cnn, leaky=leaky, batch_norm=batch_norm)

        features = init_features
        self.encoder1 = cnn_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = cnn_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = cnn_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = cnn_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = cnn_block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder4 = cnn_block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder3 = cnn_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder2 = cnn_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder1 = cnn_block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=(1, 1, 1)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


class UNet3DSimple(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=64, leaky=0.1, batch_norm=True):
        super(UNet3DSimple, self).__init__()

        cnn_block = partial(block_cnn, leaky=leaky, batch_norm=batch_norm)

        features = init_features
        self.encoder1 = cnn_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = cnn_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = cnn_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = cnn_block(features * 4, features * 8, name="bottleneck")

        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder3 = cnn_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder2 = cnn_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder1 = cnn_block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=(1, 1, 1)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


class UNet3DSimplest(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=64, leaky=0.1, batch_norm=True):
        super(UNet3DSimplest, self).__init__()

        cnn_block = partial(block_cnn, leaky=leaky, batch_norm=batch_norm)

        features = init_features
        self.encoder1 = cnn_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = cnn_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = cnn_block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder2 = cnn_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.decoder1 = cnn_block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=(1, 1, 1)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


def block_cnn(in_channels, features, name, leaky=0.1, batch_norm=True):
    if leaky > 0:
        relu_activation = nn.LeakyReLU(leaky, inplace=True)
    else:
        relu_activation = nn.ReLU(inplace=True)

    if batch_norm:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=(3, 3, 3),
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", relu_activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=(3, 3, 3),
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", relu_activation),
                ]
            )
        )
    else:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=(3, 3, 3),
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", relu_activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=(3, 3, 3),
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", relu_activation),
                ]
            )
        )


if __name__ == '__main__':
    exit()
