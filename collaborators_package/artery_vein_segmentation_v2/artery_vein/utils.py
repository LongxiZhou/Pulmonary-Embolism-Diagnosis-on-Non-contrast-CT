import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from med_transformer.file_tranfer import read_h5_file
from Tool_Functions.Functions import rotate_image


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = np.sort(os.listdir(dataset_dir))
        self.device = device

    def __getitem__(self, index):
        # print(os.path.join(self.dataset_dir, self.file_list[index]))
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))
        # print(self.file_list[index])
        # gt = read_h5_file(os.path.join(self.dataset_dir, self.file_list[index]), "label")
        raw = torch.tensor(np_array[0][np.newaxis]).to(torch.float).to(self.device)
        gt = torch.tensor(np_array[1:]).to(torch.float).to(self.device)

        return raw, gt

    def __len__(self):
        return len(self.file_list)
