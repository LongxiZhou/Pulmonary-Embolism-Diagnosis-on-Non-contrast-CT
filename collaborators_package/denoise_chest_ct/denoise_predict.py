import torch
import torch.nn as nn
from collaborators_package.denoise_chest_ct.denoise_model import UNet as DS_model
from collaborators_package.denoise_chest_ct.denoise_model import RED_CNN
import os
import numpy as np


"""
call "denoise_rescaled_array" for denoise a rescaled_ct
"""

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def load_model(show=False, model_path=None):

    if model_path is None:
        model_path = "/home/zhoul0a/Desktop/prognosis_project/check_points/denoise/denoise_model.pth"

    model = DS_model()
    if torch.cuda.device_count() > 1:
        if show:
            print("loading model from:", model_path)
            print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    if type(model) == nn.DataParallel:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    return model


def predict_denoised_red(ct_slice, device):

    h = ct_slice.shape[-1]

    ct_scan = ct_slice * 1600 - 600
    ct_scan = np.clip((ct_scan + 1000) / (3000 + 1000), 0, 1)

    ct_scan = np.transpose(ct_scan, (2, 0, 1))

    ct_scan = torch.FloatTensor(ct_scan).to(device)
    denoised_scan = np.zeros([512, 512, h])

    denoise_model = RED_CNN()
    denoise_model.load_state_dict(torch.load(
        "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/REDCNN.ckpt"))
    denoise_model = denoise_model.cuda()
    denoise_model = denoise_model.to('cuda')

    for i in range(0, h):
        raw_input = ct_scan[i:i + 1].unsqueeze(1)
        # print(raw_input.shape)
        prediction = denoise_model(raw_input).cpu().detach().numpy()[:, 0]
        denoised_scan[:, :, i:i + 1] = np.transpose(prediction, (1, 2, 0))

    return np.clip((denoised_scan * 4000 - 1000 + 600) / 1600, -0.25, 1)


def denoise_rescaled_array(np_array, model_or_model_path=None, batch_size=2):
    if model_or_model_path is None:
        model_or_model_path = load_model()
    if type(model_or_model_path) is str:
        model_or_model_path = load_model(model_path=model_or_model_path)

    model_loaded = model_or_model_path

    if len(np.shape(np_array)) == 2:
        np_array = np_array[:, :, np.newaxis]
        batch_size = 1

    num_slices = np_array.shape[-1]

    new_array = np.zeros(np.shape(np_array), 'float32')

    for i in range(0, num_slices, batch_size):

        image_pack = np_array[:, :, i: min(num_slices, i + batch_size)][np.newaxis, :]  # [1, 512, 512, batch]
        image_pack = np.swapaxes(image_pack, 0, 3)
        image_pack = np.swapaxes(image_pack, 2, 3)
        image_pack = np.swapaxes(image_pack, 1, 2)  # [batch, 1, 512, 512]

        raw_input = torch.tensor(image_pack).to(torch.float).cuda()
        prediction = model_loaded(raw_input).cpu().detach().numpy()[:, 0, :, :]  # [batch, 512, 512]

        prediction = np.swapaxes(prediction, 0, 1)
        prediction = np.swapaxes(prediction, 1, 2)

        new_array[:, :, i: min(num_slices, i + batch_size)] = prediction

    if num_slices == 1:
        new_array = new_array[:, :, 0]
    return new_array


if __name__ == '__main__':
    exit()
