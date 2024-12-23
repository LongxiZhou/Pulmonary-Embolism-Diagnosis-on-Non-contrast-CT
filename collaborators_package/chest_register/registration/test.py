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
import matplotlib.pyplot as plt
# from Artery_Vein_Segmentation.predict import predict_av
from registration_pulmonary.models.register import VxmDense as Register
from torch.nn.functional import interpolate
import visualization.visualize_3d.visualize_stl as view


def dice(array_1, array_2):
    inter = np.sum(array_1 * array_2)
    norm = np.sum(array_1 * array_1) + np.sum(array_2 * array_2)
    return 2 * inter / norm


def ceshi():
    device = torch.device('cuda:0')

    fixed_path = "/home/chuy/registration/lung_register_without_noise/384_512_320/f032_2020-03-10.npy"
    fixed = np.load(fixed_path)

    fixed_img = fixed[0][np.newaxis, np.newaxis, ...]
    fixed_img = torch.from_numpy(fixed_img).to(device).float()

    fixed_seg = fixed[1][np.newaxis, np.newaxis, ...]
    print(np.sum(fixed_seg))
    fixed_seg = torch.from_numpy(fixed_seg).to(device).float()

    moving_path = "/home/chuy/registration/lung_register_without_noise/384_512_320/f034_2020-03-10.npy"
    moving = np.load(moving_path)

    input_img = moving[0][np.newaxis, np.newaxis, ...]
    input_img = torch.from_numpy(input_img).to(device).float()

    input_seg = moving[1][np.newaxis, np.newaxis, ...]

    input_seg = torch.from_numpy(input_seg).to(device).float()

    scale_factor = 4
    vol_size = [384, 512, 320]
    nf_enc = [32, 64, 128, 256]
    nf_dec = [256, 128, 64, 64, 64, 32, 32]
    model_0 = Register(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    print("===> Loading datasets")
    model_0.load_state_dict(
        torch.load("/home/chuy/registration/checkpoint/96/mse_only/model_epoch_174.pth"))
    model_0 = model_0.cuda()
    model_0 = model_0.to('cuda:0')

    # print("===> Building model_1")
    # scale_factor = 2
    # vol_size = [384, 512, 320]
    # nf_enc = [16, 32, 64, 128]
    # nf_dec = [128, 128, 64, 64, 32, 32, 16]
    # model_1 = Register(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    # model_1.load_state_dict(
    #     torch.load("/home/chuy/registration/checkpoint/192/model_epoch_105.pth"))
    # model_1 = model_1.cuda()
    # model_1 = model_1.to('cuda:0')

    (registered_img_0, registered_seg_0, pos_flow) = model_0(input_img, fixed_img, input_seg, fixed_seg)
    # registered_seg_0[registered_seg_0 > 0.2] = 1
    # (registered_img_1, registered_seg_1, pos_flow) = model_1(registered_img_0, fixed_img, registered_seg_0, fixed_seg)
    registered_seg_0 = np.array(registered_seg_0.cpu().detach().numpy()[0, 0] > 0.25, "float32")
    registered_img_0 = registered_img_0.cpu().detach().numpy()[0, 0]
    fixed_img = fixed_img.cpu().detach().numpy()[0, 0]
    fixed_seg = fixed_seg.cpu().detach().numpy()[0, 0]
    # registered_seg_1 = np.array(registered_seg_1.cpu().detach().numpy()[0, 0] > 0.1, "float32")
    # registered_img_1 = registered_img_1.cpu().detach().numpy()[0, 0]
    for i in range(200, 205):
        plt.imshow(fixed_img[:, :, i], cmap="gray")
        plt.show()
        plt.imshow(registered_img_0[:, :, i], cmap="gray")
        plt.show()
    # print(registered_img_1.shape)
    # view.visualize_numpy_as_stl(registered_seg)
    # artery, vein = predict_av(registered_img_1, transfer=True)
    # view.visualize_two_numpy(artery, vein)
    view.visualize_two_numpy(registered_seg_0, fixed_seg)
    # view.visualize_two_numpy(registered_seg_1, fixed_seg)
    # print(dice(registered_img_0, fixed_img))
    print(dice(registered_img_0, fixed_img))
    print(np.sum(registered_seg_0))
    # print(np.sum(registered_seg_1))


ceshi()