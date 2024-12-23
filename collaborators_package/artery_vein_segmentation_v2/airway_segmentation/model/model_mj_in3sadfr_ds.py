import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .model_mj_in3sadfr import UNet3D as Base


class UNet3D(Base):
	"""
	Baseline model with Feature Recalibration module
	for pulmonary airway segmentation
	with deep supervision
	"""
	def __init__(self, in_channels=1, out_channels=1, coord=True, Dmax=80, Hmax=192, Wmax=304):
		"""
		:param in_channels: input channel numbers
		:param out_channels: output channel numbers
		:param coord: boolean, True=Use coordinates as position information, False=not
		:param Dmax: the size of the largest feature cube in depth, default=80
		:param Hmax: the size of the largest feature cube in height, default=192
		:param Wmax: the size of the largest feature cube in width, default=304
		"""
		super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, coord=coord,\
		                             Dmax=Dmax, Hmax=Hmax, Wmax=Wmax)
		self.upsampling4 = nn.Upsample(scale_factor=4)
		self.upsampling8 = nn.Upsample(scale_factor=8)
		self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1)  # deep supervision
		self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1)  # deep supervision
		self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1)  # deep supervision

	def forward(self, input, coordmap=None):
		"""
		:param input: shape = (batch_size, num_channels, D, H, W) \
		:param coordmap: shape = (batch_size, 3, D, H, W)
		:return: output segmentation tensors list, attention mapping
		"""
		conv1 = self.conv1(input)
		conv1, _ = self.pe1(conv1)
		x = self.pooling(conv1)
		
		conv2 = self.conv2(x)
		conv2, _ = self.pe2(conv2)
		x = self.pooling(conv2)
		
		conv3 = self.conv3(x)
		conv3, mapping3 = self.pe3(conv3)
		x = self.pooling(conv3)
		
		conv4 = self.conv4(x)
		conv4, mapping4 = self.pe4(conv4)
		x = self.pooling(conv4)

		conv5 = self.conv5(x)
		conv5, mapping5 = self.pe5(conv5)

		x = self.upsampling(conv5)
		x = torch.cat([x, conv4], dim=1)
		conv6 = self.conv6(x)
		conv6, mapping6 = self.pe6(conv6)
		ds_6 = self.sigmoid(self.upsampling8(self.dsconv6(conv6)))
		
		x = self.upsampling(conv6)
		x = torch.cat([x, conv3], dim=1)
		conv7 = self.conv7(x)
		conv7, mapping7 = self.pe7(conv7)
		ds_7 = self.sigmoid(self.upsampling4(self.dsconv7(conv7)))
		
		x = self.upsampling(conv7)
		x = torch.cat([x, conv2], dim=1)
		conv8 = self.conv8(x)
		conv8, mapping8 = self.pe8(conv8)
		ds_8 = self.sigmoid(self.upsampling(self.dsconv8(conv8)))
		
		x = self.upsampling(conv8)

		if (self._coord is True) and (coordmap is not None):
			x = torch.cat([x, conv1, coordmap], dim=1)
		else:
			x = torch.cat([x, conv1], dim=1)

		conv9 = self.conv9(x)
		conv9, mapping9 = self.pe9(conv9)

		x = self.conv10(conv9)

		x = self.sigmoid(x)

		return [x, ds_6, ds_7, ds_8], [mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]


if __name__ == '__main__':
	net = UNet3D(in_channels=1, out_channels=1, coord=False)
	print(net)
	print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
# Number of network parameters: 4237283  Baseline + Feature Recalibration + Deep Supervision
