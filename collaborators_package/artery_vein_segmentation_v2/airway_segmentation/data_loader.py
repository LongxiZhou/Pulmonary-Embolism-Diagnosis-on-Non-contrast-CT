import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
from utils import load_itk_image, lumTrans

class AirwayData(Dataset):
	"""
	Generate dataloader
	"""
	def __init__(self, split_comber=None, data_dir=None):
		"""
		:param split_comber: split-combination-er
		:param data_dir: input CT image directory, nifty format
		"""
		self.split_comber = split_comber
		self.raw_path = data_dir
		# specify the paths
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}

		assert(os.path.exists(self.raw_path) is True)
		imgs, origin, spacing = load_itk_image(self.raw_path)
		if np.amax(imgs) > 255 or np.amin(imgs) < 0:
			imgs = lumTrans(imgs)
		splits, nzhw, orgshape = self.split_comber.split_id(imgs)
		data_name = self.raw_path.split('/')[-1].split('.nii')[0]
		# print ("Name: %s, # of splits: %d"%(data_name, len(splits)))
		allimgdata_memory[data_name] = [imgs, origin, spacing]
		for j in range(len(splits)):
			cursplit = splits[j]
			curlist = [data_name, cursplit, j, nzhw, orgshape]
			cubelist.append(curlist)

		self.allimgdata_memory = allimgdata_memory
		random.shuffle(cubelist)
		self.cubelist = cubelist
		# print ('total cubelist number: %d'%(len(self.cubelist)))

	def __len__(self):
		"""
		:return: length of the dataset
		"""
		return len(self.cubelist)

	def __getitem__(self, idx):
		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t%1)[2:7]))  # seed according to time
		curlist = self.cubelist[idx]
		curNameID = curlist[0]
		cursplit = curlist[1]
		curSplitID = curlist[2]
		curnzhw = curlist[3]
		curShapeOrg = curlist[4]
		####################################################################
		imginfo = self.allimgdata_memory[curNameID]
		imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
		####################################################################
		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		curcube = (curcube.astype(np.float32))/255.0

		####################################################################
		###calculate the coordinate for coordinate-aware convolution########
		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
		normstart = ((np.array(start).astype('float')/np.array(curShapeOrg).astype('float'))-0.5)*2.0
		crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]
		stride = 1.0
		normsize = (np.array(crop_size).astype('float')/np.array(curShapeOrg).astype('float'))*2.0
		xx, yy, zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(crop_size[0])),
							   np.linspace(normstart[1],normstart[1]+normsize[1],int(crop_size[1])),
							   np.linspace(normstart[2],normstart[2]+normsize[2],int(crop_size[2])),
							   indexing ='ij')
		coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...], zz[np.newaxis,:]], 0).astype('float')
		assert (coord.shape[0] == 3)
		####################################################################
		curNameID = [curNameID]
		curSplitID = [curSplitID]
		curnzhw = np.array(curnzhw)
		curShapeOrg = np.array(curShapeOrg)
		curcube = curcube[np.newaxis,...]
		####################################################################
		return torch.from_numpy(curcube).float(), torch.from_numpy(coord).float(),\
		torch.from_numpy(origin), torch.from_numpy(spacing), curNameID, curSplitID,\
		torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)