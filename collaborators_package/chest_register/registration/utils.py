import glob
import os
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.functional import interpolate
import numpy as np
from End_to_end_Segmentation.training.ssim_loss import ssim


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = glob.glob((os.path.join(dataset_dir, "*.npy")))
        self.device = device

    def __getitem__(self, index):
        # print(self.file_list[index])
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))
        # print(np_array.shape)
        raw = torch.tensor(np_array[0:1]).to(self.device).to(torch.float)
        raw[raw > 1] = 1
        raw[raw < -0.25] = -0.25
        mask = torch.tensor(np_array[1:2]).to(self.device).to(torch.float)
        return [raw, mask]

    def __len__(self):
        return len(self.file_list)
