import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class VoxUNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=3):
        super(VoxUNet, self).__init__()

        self.encoder1_1 = nn.Conv3d(in_channel, 16, 3, stride=1, padding=1)
        self.encoder1_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 128, 128

        self.encoder2_1 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 64, 64

        self.encoder3_1 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 32, 32

        self.encoder4_1 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4_2 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 16, 16

        self.decoder4_1 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.decoder4_2 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.map4 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=16, mode='trilinear')
        )
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.decoder3_1 = nn.Conv3d(128 + 128, 64, 3, stride=1, padding=1)
        self.decoder3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=8, mode='trilinear')
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.decoder2_1 = nn.Conv3d(64 + 64, 32, 3, stride=1, padding=1)
        self.decoder2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.map2 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=4, mode='trilinear')
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.decoder1_1 = nn.Conv3d(32 + 32, 16, 3, stride=1, padding=1)
        self.decoder1_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.map1 = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.map0 = nn.Sequential(
            nn.Conv3d(16 + 16, 16, 1, 1),
            nn.Conv3d(16, out_channel, 1, 1)
        )

        self.down = nn.MaxPool3d(kernel_size=4, stride=4)
        self.transfer = SpatialTransformer(size=[256, 256, 256])
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, src, tgt, feature, fixed_feature):
        x = torch.cat([src, tgt, feature, fixed_feature], dim=1)  # 2, 256, 256, 256

        x = self.encoder1_1(x)
        en_1 = self.encoder1_2(x)
        x = self.pool1(en_1)
        x = self.relu(x)

        x = self.encoder2_1(x)
        en_2 = self.encoder2_2(x)  # 2, 128, 128, 128
        x = self.pool2(en_2)
        x = self.relu(x)

        x = self.encoder3_1(x)
        en_3 = self.encoder3_2(x)  # 2, 64, 64, 64
        x = self.pool3(en_3)
        x = self.relu(x)

        x = self.encoder4_1(x)
        en_4 = self.encoder4_2(x)  # 2, 32, 32, 32
        x = self.pool4(en_4)
        x = self.relu(x)

        # 2, 16, 16, 16
        x = self.decoder4_1(x)
        x = self.decoder4_2(x)
        mapping_4 = self.map4(x)  # 3, 16, 16, 16
        x = self.up4(x)
        x = self.relu(x)

        x = self.decoder3_1(torch.cat([x, en_4], dim=1))
        x = self.decoder3_2(x)
        mapping_3 = self.map3(x)  # 3, 32, 32, 32
        x = self.up3(x)
        x = self.relu(x)

        x = self.decoder2_1(torch.cat([x, en_3], dim=1))
        x = self.decoder2_2(x)
        mapping_2 = self.map2(x)  # 3, 64, 64, 64
        x = self.up2(x)
        x = self.relu(x)

        x = self.decoder1_1(torch.cat([x, en_2], dim=1))
        x = self.decoder1_2(x)
        mapping_1 = self.map1(x)  # 3, 128, 128, 128
        x = self.up1(x)
        x = self.relu(x)

        mapping_0 = self.map0(torch.cat([x, en_1], dim=1))

        ori_4 = self.transfer(src, mapping_4)
        ori_3 = self.transfer(ori_4, mapping_3)
        ori_2 = self.transfer(ori_3, mapping_2)
        ori_1 = self.transfer(ori_2, mapping_1)
        ori_final = self.transfer(ori_1, mapping_0)

        # print(mapping_4.shape)
        feature_4 = self.transfer(feature, mapping_4)
        feature_3 = self.transfer(feature_4, mapping_3)
        feature_2 = self.transfer(feature_3, mapping_2)
        feature_1 = self.transfer(feature_2, mapping_1)
        feature_final = self.transfer(feature_1, mapping_0)

        return ori_final, feature_final


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
