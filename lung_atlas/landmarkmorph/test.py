# python imports
import os
import glob
# external imports
import torch
import numpy as np

import SimpleITK as sitk
# internal imports
from lung_atlas.landmarkmorph import losses
from lung_atlas.landmarkmorph.config import args
from lung_atlas.landmarkmorph.model import UNetwork, SpatialTransformer
from scipy.ndimage import zoom


def compress(array):
    num = 512
    new_array = np.zeros([256, 256, 256])
    for i in range(int(num / 2 - 1)):
        for j in range(int(num / 2 - 1)):
            for k in range(int(num / 2 - 1)):
                new_array[i, j, k] = max(array[2 * i, 2 * j, 2 * k],
                                         array[2 * i + 1, 2 * j, 2 * k],
                                         array[2 * i, 2 * j + 1, 2 * k],
                                         array[2 * i, 2 * j, 2 * k + 1],
                                         array[2 * i + 1, 2 * j + 1, 2 * k],
                                         array[2 * i + 1, 2 * j, 2 * k + 1],
                                         array[2 * i, 2 * j + 1, 2 * k + 1],
                                         array[2 * i + 1, 2 * j + 1, 2 * k + 1])
    return new_array


def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    fixed = np.load(args.atlas_file)
    f_img = fixed[:, :, :, 0]
    input_fixed = f_img[np.newaxis, np.newaxis, :]
    vol_size = input_fixed.shape[2:]
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    print(vol_size)

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.npy"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model_guided and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = UNetwork(3, nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN_img = SpatialTransformer(vol_size).to(device)
    UNet.eval()
    STN_img.eval()

    for file in test_file_lst:
        name = os.path.split(file)[1]
        print(name)
        # 读入moving图像
        input_img = np.load(os.path.join(args.test_dir, name))

        # data processing
        raw = input_img[0]
        lung = input_img[1]
        blood_vessel = input_img[2]
        airway = input_img[3]
        heart = input_img[4]

        # compress
        raw = zoom(raw, 0.25)[np.newaxis, np.newaxis, ...]
        # print(raw.shape)
        lung = zoom(lung, 0.25)[np.newaxis, np.newaxis, ...]
        blood_vessel = zoom(blood_vessel, 0.25)[np.newaxis, np.newaxis, ...]
        airway = zoom(airway, 0.25)[np.newaxis, np.newaxis, ...]
        heart = zoom(heart, 0.25)[np.newaxis, np.newaxis, ...]

        # to gpu
        raw = torch.from_numpy(raw).to(device).float()
        lung = torch.from_numpy(lung).to(device).float()
        blood_vessel = torch.from_numpy(blood_vessel).to(device).float()
        airway = torch.from_numpy(airway).to(device).float()
        heart = torch.from_numpy(heart).to(device).float()

        # 获得配准后的图像和label
        pre_flow = UNet(raw, input_fixed)

        pre_raw = STN_img(raw, pre_flow)
        pre_lung = STN_img(lung, pre_flow)
        pre_blood = STN_img(blood_vessel, pre_flow)
        pre_airway = STN_img(airway, pre_flow)
        pre_heart = STN_img(heart, pre_flow)

        pre = torch.zeros([5, 128, 128, 128], dtype=torch.float)
        pre[0] = pre_raw
        pre[1] = pre_lung
        pre[2] = pre_blood
        pre[3] = pre_airway
        pre[4] = pre_heart

        pre = pre.cpu().detach().numpy()
        np.save("/home/chuy/registration/128*128*128/pre_allseg/" + name, pre)


if __name__ == "__main__":
    test()
