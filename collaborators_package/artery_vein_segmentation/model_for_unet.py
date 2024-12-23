import importlib
import torch.nn as nn
import torch
import torch.nn.functional as F
from .buildingblocks import Encoder, Decoder, DoubleConv, ExtResNetBlock
from .utils_torch import number_of_features_per_level


class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                # currently pools with a constant kernel: (2, 2, 2)
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=basic_module,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order,
                                             num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             **kwargs)


class CBRConv_3d(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dirate=1):
        super(CBRConv_3d, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3),
                              stride=(1, 1, 1), padding=1*dirate, dilation=(dirate, dirate, dirate))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CBRConv_2d(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dirate=1):
        super(CBRConv_2d, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 1),
                              stride=(1, 1, 1), padding=(1, 1, 0), dilation=(1, 1, dirate))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def upsample(src, scale=(2, 2, 2)):
    src = nn.Upsample(src, scale_factor=scale, mode='trilinear')
    return src


class RSU7(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU7, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_6 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_7 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        self.re_conv_6 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_5 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx = self.pool(hx_3)

        hx_4 = self.conv_4(hx)
        hx = self.pool(hx_4)

        hx_5 = self.conv_5(hx)
        hx = self.pool(hx_5)

        hx_6 = self.conv_6(hx)
        hx_7 = self.conv_7(hx_6)

        hx_6d = self.re_conv_6(torch.cat((hx_7, hx_6), 1))
        hx_6d_up = self.upsample(hx_6d)

        hx_5d = self.re_conv_5(torch.cat((hx_6d_up, hx_5), 1))
        hx_5d_up = self.upsample(hx_5d)

        hx_4d = self.re_conv_4(torch.cat((hx_5d_up, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU6(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU6, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_6 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_5 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx = self.pool(hx_3)

        hx_4 = self.conv_4(hx)
        hx = self.pool(hx_4)

        hx_5 = self.conv_5(hx)
        hx_6 = self.conv_6(hx_5)

        hx_5d = self.re_conv_5(torch.cat((hx_6, hx_5), 1))
        hx_5d_up = self.upsample(hx_5d)

        hx_4d = self.re_conv_4(torch.cat((hx_5d_up, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU5(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU5, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)

        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx = self.pool(hx_3)

        hx_4 = self.conv_4(hx)
        hx_5 = self.conv_5(hx_4)

        hx_4d = self.re_conv_4(torch.cat((hx_5, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU4(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU4, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)

        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx_4 = self.conv_4(hx_3)

        hx_3d = self.re_conv_3(torch.cat((hx_4, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU4F(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU4F, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=2)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=4)
        self.conv_4 = CBRConv_3d(mid_channel, mid_channel, dirate=8)

        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=4)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=2)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx_2 = self.conv_2(hx_1)
        hx_3 = self.conv_3(hx_2)
        hx_4 = self.conv_4(hx_3)

        hx_3d = self.re_conv_3(torch.cat((hx_4, hx_3), 1))
        hx_2d = self.re_conv_2(torch.cat((hx_3d, hx_2), 1))
        hx_1d = self.re_conv_1(torch.cat((hx_2d, hx_1), 1))

        return hx_1d + hx_in


class U2NET(nn.Module):

    def __init__(self, in_ch=1, out_ch=3):
        super(U2NET, self).__init__()

        channel = 8
        self.stage1 = RSU7(in_ch, channel, channel*2)
        self.stage2 = RSU6(channel*2, channel, channel*4)
        self.stage3 = RSU5(channel*4, channel*2, channel*8)
        self.stage4 = RSU4(channel*8, channel*4, channel*16)
        self.stage5 = RSU4F(channel*16, channel*8, channel*16)
        self.stage6 = RSU4F(channel*16, channel*8, channel*16)

        # decoder
        self.stage5d = RSU4F(channel*32, channel*8, channel*16)
        self.stage4d = RSU4(channel*32, channel*4, channel*8)
        self.stage3d = RSU5(channel*16, channel*2, channel*4)
        self.stage2d = RSU6(channel*8, channel, channel*2)
        self.stage1d = RSU7(channel*4, 16, channel*2)

        self.side1 = nn.Conv3d(channel*2, out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(channel*2, out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(channel*4, out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(channel*8, out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(channel*16, out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(channel*16, out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6 * out_ch, out_ch, 1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.upsample_3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upsample_4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upsample_5 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')
        self.upsample_6 = nn.Upsample(scale_factor=(32, 32, 32), mode='trilinear')

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool(hx5)

        hx6 = self.stage6(hx)
        hx6up = self.upsample(hx6)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upsample(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upsample(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upsample(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upsample(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = self.upsample_2(d2)

        # print(hx3d.shape)
        d3 = self.side3(hx3d)
        d3 = self.upsample_3(d3)

        # print(hx4d.shape)
        d4 = self.side4(hx4d)
        d4 = self.upsample_4(d4)

        d5 = self.side5(hx5d)
        d5 = self.upsample_5(d5)

        d6 = self.side6(hx6)
        d6 = self.upsample_6(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        # return F.sigmoid(d0)
        return F.sigmoid(d0), [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)]


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('Artery_Vein_Segmentation.model_for_unet')
        clazz = getattr(m, class_name)
        return clazz

    model_config = config
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
