import numpy as np
import os
from analysis.connectivity_yuetan import select_region
from filter.vessel_2d import vessel_enhance_xyz
from scipy.ndimage import zoom
from Artery_Vein_Segmentation.predict import predict_airway
import torch
from semantic_segmentation.artery_vein.model import UNet
from visualization.visualize_3d import visualize_stl as view

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


raw_path = "/home/chuy/Desktop/CTA/rescaled_ct"
airway_path = "/data/chest_CT/rescaled_ct/non-contrast/ground_truth/airway_gt"
lung_path = "/home/chuy/Desktop/CTA/semantics/lung_mask"
artery_path = "/data/chest_CT/rescaled_ct/non-contrast/ground_truth/artery_gt"
# artery_2_path = "/home/chuy/Train_and_Test/Artery_Vein_Upsampling/mask/artery_2"
#
vein_path = "/data/chest_CT/rescaled_ct/non-contrast/ground_truth/vein_gt"
# vein_2_path = "/home/chuy/Train_and_Test/Artery_Vein_Upsampling/mask/vein_2"

h5_file = "/data/Train_and_Test/segmentation/artery_vein"


def preprocess(array):
    array = array[np.newaxis, np.newaxis]
    array = torch.from_numpy(array)
    array = array.to(device)
    array = array.to(torch.float32)
    return array


def pre_av(raw_array):
    model = UNet(in_channel=1, num_classes=2)
    model.load_state_dict(torch.load(
        "/data/Train_and_Test/segmentation/new_model/model_epoch_1.pth"))
    model = model.cuda()
    model = model.to('cuda')
    raw_array = preprocess(raw_array)
    pre = model(raw_array).detach().cpu().numpy()

    pre_artery = np.array(pre[0, 0] > 0.5, "float32")
    pre_vein = np.array(pre[0, 1] > 0.5, "float32")
    return pre_artery, pre_vein


for filename in os.listdir(raw_path):
    print(filename)
    raw = np.load(os.path.join(raw_path, filename))["array"]
    lung = np.load(os.path.join(lung_path, filename.replace('npy', "npz")))["array"]
    # artery_1 = np.load(os.path.join(artery_path, filename.replace('npy', "npz")))["array"]
    # vein_1 = np.load(os.path.join(vein_path, filename.replace('npy', "npz")))["array"]

    raw = np.clip(raw, -0.25, 0.75) + 0.25
    # artery_1 *= np.array(raw > 0.2, "float32")
    # vein_1 *= np.array(raw > 0.2, "float32")

    zoomed_raw = np.clip(zoom(raw, 0.5), 0, 1)
    artery, vein = pre_av(zoomed_raw)
    # artery = np.array(zoom(artery_1[:, :, 128:], 0.5, order=2) > 0.5, "float32")
    # vein = np.array(zoom(vein_1[:, :, 128:], 0.5, order=2) > 0.5, "float32")
    view.visualize_two_numpy(artery, vein)

    artery = zoom(artery, 2, order=0)
    vein = zoom(vein, 2, order=0)
    # convert_point_to_surf(select_region(artery, num=1))
    # convert_point_to_surf(select_region(vein, num=1))

    view.visualize_two_numpy(artery, vein)

    # blood = np.array(vessel_enhance_xyz(raw, lung) + artery + vein > 0.5, "float32")
    # view.visualize_numpy_as_stl(blood)
    # artery, vein = refinement(raw, artery, vein, blood)
    # artery_2, vein_2 = predict_av_2(raw - 0.25)
    # artery = select_region(np.array(artery + artery_2 > 0.5, "float32"), 1)
    # vein += select_region(np.array(vein + vein_2 > 0.5, "float32"), 1)
    # view.visualize_numpy_as_stl(artery)
    # view.visualize_numpy_as_stl(vein)

    # vessel = vessel_enhance(raw, lung=lung)

    # view.visualize_two_numpy(artery, vessel)
    # view.visualize_two_numpy(vein, vessel)
    airway = predict_airway(raw, transfer=True)
    view.visualize_numpy_as_stl(airway)
    airway_filter = vessel_enhance_xyz(raw, lung)
    view.visualize_numpy_as_stl(airway_filter)
    airway = np.array(airway_filter + airway > 0.5, "float32")
    airway = select_region(airway, 1)
    view.visualize_numpy_as_stl(airway)