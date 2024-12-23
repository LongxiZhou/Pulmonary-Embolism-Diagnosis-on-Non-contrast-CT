import torch.nn as nn
import torch


class encoding_block(nn.Module):
    """
    Convolutional batch norm_layer block with relu activation (main block used in the encoding steps)
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout=False,
    ):
        super().__init__()

        if batch_norm:

            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding,
                          stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=(1, 1)),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(2, 2), stride=(2, 2))

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.upsample(input1, output2.size()[2:], mode="bilinear")

        return self.conv(torch.cat([output1, output2], 1))


class UNet(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, in_channel=1, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(in_channel, 64)
        self.maxpool1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = encoding_block(64, 128)
        self.maxpool2 = nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = encoding_block(128, 256)
        self.maxpool3 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = encoding_block(256, 512)
        self.maxpool4 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

        # center
        self.center = encoding_block(512, 1024)

        # decoding
        self.decode4 = decoding_block(1024, 512, upsampling=True)
        self.decode3 = decoding_block(512, 256, upsampling=True)
        self.decode2 = decoding_block(256, 128, upsampling=True)
        self.decode1 = decoding_block(128, 64, upsampling=False)

        # final
        self.final = nn.Conv2d(64, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        # print(conv1.shape)
        maxpool1 = self.maxpool1(conv1)
        # print(maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final = self.final(decode1)

        return final


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class UNet_2(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, in_channel=1, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(in_channel, 64, kernel_size=5, padding=0)
        self.maxpool1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = encoding_block(64, 128, kernel_size=5, padding=0)
        self.maxpool2 = nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = encoding_block(128, 256, kernel_size=5, padding=0)
        self.maxpool3 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = encoding_block(256, 512, kernel_size=5, padding=0)
        self.maxpool4 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

        # center
        self.center = encoding_block(512, 1024)

        # decoding
        self.decode4 = decoding_block(1024, 512, upsampling=True)
        self.decode3 = decoding_block(512, 256, upsampling=True)
        self.decode2 = decoding_block(256, 128, upsampling=True)
        self.decode1 = decoding_block(128, 64, upsampling=False)

        # final
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, input):
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)
        decode3 = self.decode3(conv3, decode4)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)

        # final
        final = input + self.final(decode1)

        return final


class UNet_2303(nn.Module):
    def __init__(self, out_ch=64):
        super(UNet_2303, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch * 2, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch * 4, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch * 2, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
