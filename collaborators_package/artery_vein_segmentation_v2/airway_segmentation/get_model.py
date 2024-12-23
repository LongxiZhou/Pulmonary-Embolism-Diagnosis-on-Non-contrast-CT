import torch
import torch.nn as nn
import numpy as np
from semantic_segmentation.airway_segmentation.model.model_mj_in3sadfr import UNet3D
# Baseline + Feature Recalibration

config = {'pad_value': 0,
          'augtype': {'flip': True, 'swap': False, 'smooth': False, 'jitter': True, 'split_jitter': True},
          'startepoch': 0, 'lr_stage': np.array([10, 20, 40, 60]), 'lr': np.array([3e-3, 3e-4, 3e-5, 3e-6]),
          'dataset_path': 'preprocessed_datasets', 'dataset_split': './split_dataset.pickle'}


def get_model(cubesize):
    net = UNet3D(in_channels=1, out_channels=1, coord=True,
                 Dmax=cubesize[0], Hmax=cubesize[1], Wmax=cubesize[2])
    return net
