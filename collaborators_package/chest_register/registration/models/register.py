import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collaborators_package.chest_register.registration.models import layers


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 4 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] + enc_nf[-2], dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] + enc_nf[-3], dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 4, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.1))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.1))
        return layer

    def forward(self, src, tgt, seg, fixed_seg):
        x = torch.cat([src, tgt, seg, fixed_seg], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            # print(x.shape)
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            # print(y.shape)
            y = self.dec[i](y)
            # print(y.shape)
            y = self.upsample(y)
            # print(y.shape)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            # print(y.shape, x_enc[0].shape)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
                 inshape,
                 unet_encoder=None,
                 unet_decoder=None,
                 scale=1,
                 ndims=3,
                 int_steps=7,
                 bidir=False,
                 use_morph=True):
        super().__init__()
        self.training = True
        self.scale = scale
        self.unet_model = U_Network(ndims, unet_encoder, unet_decoder, bn=False)

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(ndims, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.bidir = bidir
        if use_morph:
            self.integrate = layers.VecInt(inshape, int_steps)
        else:
            self.integrate = None

        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, src, tgt, src_seg, tgt_seg):
        # concatenate inputs and propagate unet=
        mode = "trilinear"
        if self.scale != 1:

            src_rescaled = F.interpolate(src, scale_factor=1 / self.scale, mode=mode, align_corners=True)
            tgt_rescaled = F.interpolate(tgt, scale_factor=1 / self.scale, mode=mode, align_corners=True)
            src_seg_rescaled = F.interpolate(src_seg, scale_factor=1 / self.scale, mode=mode, align_corners=True)
            tgt_seg_rescaled = F.interpolate(tgt_seg, scale_factor=1 / self.scale, mode=mode, align_corners=True)

            x = self.unet_model(src_rescaled, tgt_rescaled, src_seg_rescaled, tgt_seg_rescaled)
        else:
            x = self.unet_model(src, tgt, src_seg, tgt_seg)
        # transform into flow field
        flow_field_0 = self.flow(x)
        if self.scale != 1:
            flow_field = F.interpolate(flow_field_0 * self.scale, scale_factor=self.scale)
        else:
            flow_field = flow_field_0

        # resize flow for integration
        pos_flow = flow_field
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

        # warp image with flow field
        src_re = self.transformer(src, pos_flow)
        src_seg_re = self.transformer(src_seg, pos_flow)
        tgt_re = self.transformer(tgt, neg_flow) if self.bidir else None
        tgt_seg_re = self.transformer(tgt_seg, neg_flow) if self.bidir else None

        return (src_re, src_seg_re, tgt_re, tgt_seg_re, pos_flow) if self.bidir else (src_re, src_seg_re, flow_field_0)


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

