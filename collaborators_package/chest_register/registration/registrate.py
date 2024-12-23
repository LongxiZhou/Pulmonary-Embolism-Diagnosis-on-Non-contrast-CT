# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
import torch.nn as nn
from registration_pulmonary.models import register_to_standard, register_96, VoxUNet_1
from torch.nn.functional import interpolate


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def preprocess(array, device):
    array = array[np.newaxis, np.newaxis, :]
    array = torch.from_numpy(array)
    array = array.to(device)
    array = array.to(torch.float)
    return array


def do_registration(input_img, fixed_img, input_seg, fixed_seg, transfer=False, down_sampling=False, inter=False):
    """
    :param input_img: input scan with shape 256*256*256 or 512*512*512 (need down_sampling=Trye)
    :param fixed_img:
    :param input_seg:
    :param fixed_seg:
    :param transfer: whether transfer from altolia to zhou
    :param down_sampling: 512->256
    :param inter: register to standard or register interactively
    :return: the registered img and seg. The shape is the same with input
    """
    device = torch.device('cuda:0')
    vol_size = [128, 128, 128]
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]

    model = register_interact.U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    spatial_transfer = register_interact.SpatialTransformer(vol_size).to(device)
    # model = VoxUNet_1.VoxUNet().to(device)
    checkpoint = torch.load("/home/chuy/PycharmProjects/VoxelMorph-torch-master/Checkpoint/3400.pth")
    model.load_state_dict(checkpoint)
    model.train()

    # model = VoxUNet_1.VoxUNet().to(device)
    # spatial_transfer = register_interact.SpatialTransformer(vol_size).to(device)
    # checkpoint = torch.load("/home/chuy/registration/new_inter_model/2000.pth")
    # model.load_state_dict(checkpoint)
    # model.train()

    # print("UNet: ", count_parameters(model))
    # print(model)

    if transfer:
        input_img = (input_img * 1600 + 400) / 1400
        fixed_img = (fixed_img * 1600 + 400) / 1400

    input_seg = preprocess(input_seg, device)
    input_img = preprocess(input_img, device)
    fixed_seg = preprocess(fixed_seg, device)
    fixed_img = preprocess(fixed_img, device)

    mode = "trilinear"
    if down_sampling:
        input_img_down = interpolate(input_img, scale_factor=0.25, mode=mode, align_corners=True)
        fixed_img_down = interpolate(fixed_img, scale_factor=0.25, mode=mode, align_corners=True)
        input_seg_down = interpolate(input_seg, scale_factor=0.25, mode=mode, align_corners=True)
        fixed_seg_down = interpolate(fixed_seg, scale_factor=0.25, mode=mode, align_corners=True)
        flow = model(input_img_down, fixed_img_down, input_seg_down)
        flow = interpolate(flow, scale_factor=4, mode=mode, align_corners=True)
        # registered_img, registered_seg = model(input_img, fixed_img, input_seg, fixed_seg)
        # registered_img = interpolate(registered_img, scale_factor=2, mode=mode, align_corners=True)
        # registered_seg = interpolate(registered_seg, scale_factor=2, mode=mode, align_corners=True)
    else:
        flow = model(input_img, fixed_img, input_seg)
        registered_img, registered_seg = model(input_img, fixed_img, input_seg, fixed_seg)

    registered_img = spatial_transfer(input_img, flow)
    registered_seg = spatial_transfer(input_seg, flow)

    registered_img = registered_img.detach().cpu().numpy()
    registered_seg = registered_seg.detach().cpu().numpy()
    # flow = flow.detach().cpu().numpy()

    if transfer:
        registered_img = (input_img * 1400 - 400) / 1600

    return registered_img[0, 0], registered_seg[0, 0]
