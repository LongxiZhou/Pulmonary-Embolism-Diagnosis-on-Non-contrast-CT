import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from Tool_Functions.Functions import rotate_image


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = np.sort(os.listdir(dataset_dir))
        self.device = device

    def __getitem__(self, index):
        # print(os.path.join(self.dataset_dir, self.file_list[index]))
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"]
        raw = torch.tensor(np_array[0:1]).to(torch.float).to(self.device)
        reference = torch.tensor(np_array[1:2]).to(torch.float).to(self.device)
        coord = torch.tensor(np_array[2:]).to(torch.float).to(self.device)

        return raw, reference, coord

    def __len__(self):
        return len(self.file_list)