import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from semantic_segmentation.artery_vein.utils import TrainSetLoader
from semantic_segmentation.artery_vein.model import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


def predict(raw_file, airway, blood):
    raw = np.clip(raw_file, -0.25, 0.75) + 0.25
    model = UNet(in_channel=5, num_classes=2)
    model.load_state_dict(torch.load(
            "/data/Train_and_Test/segmentation/artery_vein_model/model_epoch_7.pth"))
    model = model.to('cuda')

    artery = np.zeros([512, 512, 512])
    vein = np.zeros([512, 512, 512])
    for i in range(100, 450):
        training = np.zeros([512, 512, 5])
        training[:, :, 0:3] = 0.5 * artery[:, :, i - 3:i] + 0.75 * vein[:, :, i - 3:i]
        training[:, :, 3] = raw[i]
        training[:, :, 4] = 0.25 * airway[:, :, i] + 0.75 * blood[:, :, i]
        training = np.transpose(training, (2, 0, 1))
        training = torch.tensor(training[np.newaxis, :]).to(torch.float).to(device)

        pre = model(training).detach().cpu().numpy()
        artery[:, :, i] = np.array(pre[0, 0] > 0.5, "float32")
        vein[:, :, i] = np.array(pre[0, 1] > 0.5, "float32")

    return artery, vein


