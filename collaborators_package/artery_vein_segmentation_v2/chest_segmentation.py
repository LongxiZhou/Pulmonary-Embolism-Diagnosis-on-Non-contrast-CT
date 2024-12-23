import numpy as np
import torch
from analysis.center_line_and_depth_3D import get_center_line
from analysis.connectivity_yuetan import select_region
from collaborators_package.denoise_chest_ct.denoise_predict import predict_denoised_red
from torch.utils.data import DataLoader
from format_convert.spatial_normalize import rescale_to_new_shape
from collaborators_package.artery_vein_segmentation.model_for_unet import get_model, U2NET
from collaborators_package.artery_vein_segmentation.utils_torch import load_checkpoint
from torch.utils.data.dataset import Dataset
from collaborators_package.artery_vein_segmentation_v2.artery_vein.model import UNet as UNet_av
import os
from collaborators_package.do_filter.frangi_3d import vessel_enhancement
from visualization.visualize_3d import visualize_stl as view
from collaborators_package.artery_vein_segmentation_v2.artery_vein.artery_vein_refine import refinement
from basic_tissue_prediction.predict_rescaled import predict_lung_masks_rescaled_array


def preprocess(pre_array, device, tissue="airway"):
    array = pre_array[np.newaxis]
    if tissue == "av":
        array = array[np.newaxis]
    array = torch.from_numpy(array)
    array = array.to(device)
    array = array.to(torch.float16)
    return array


def sigmoid(img, alpha, beta):
    return 1 / (1 + np.exp((beta - img) / alpha))


class TestSetLoader(Dataset):
    def __init__(self, slices, raw_array, device):
        super(TestSetLoader, self).__init__()
        self.slice = slices
        self.raw = raw_array
        self.device = device

    def __getitem__(self, index):
        area = self.slice[index]
        x = area[0]
        y = area[1]
        z = area[2]
        sub_array = self.raw[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        area = np.array([x[0], x[1], y[0], y[1], z[0], z[1]])
        return area, preprocess(sub_array, self.device)

    def __len__(self):
        return len(self.slice)


def array_cut(np_array, patch, stride):
    def gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    i_z, i_y, i_x = np_array.shape
    [k_z, k_y, k_x] = patch
    [s_z, s_y, s_x] = stride
    z_steps = gen_indices(i_z, k_z, s_z)
    slices = []
    for z in z_steps:
        y_steps = gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = [
                    [z, z + k_z],
                    [y, y + k_y],
                    [x, x + k_x]
                ]
                slices.append(slice_idx)

    return slices


def predict_one_stage(test_loader, model, shape, avg, tissue="airway"):

    if tissue == "airway":
        prediction = np.zeros([1, shape[0], shape[1], shape[2]])
    elif tissue == "av":
        prediction = np.zeros([2, shape[0], shape[1], shape[2]])
    else:
        return None

    for iteration, (area, sub_array) in enumerate(test_loader):
        if torch.sum(sub_array) == 0:
            # print("pass")
            continue
        with torch.no_grad():
            predict_result = model(sub_array)[0].detach().cpu().numpy()
        # print(predict_result.shape)
        if tissue == "airway":
            prediction[:, area[0, 0]:area[0, 1], area[0, 2]:area[0, 3], area[0, 4]:area[0, 5]] \
                += predict_result[0]
        elif tissue == "av":
            prediction[:, area[0, 0]:area[0, 1], area[0, 2]:area[0, 3], area[0, 4]:area[0, 5]] \
                += predict_result

    return prediction / avg


def judge_blood_quality(blood_mask, show=False):
    central_line = get_center_line(blood_mask)
    print(np.sum(central_line))

    if np.sum(central_line) < 7000:
        if show:
            view.visualize_numpy_as_stl(blood_mask)
        return True
    else:
        return False


def predict_intra_av_2(scan, device):
    patch = [512, 512, 32]
    stride = [512, 512, 8]

    config_model = {
        "f_maps": 32,
        "final_sigmoid": False,
        "in_channels": 1,
        "out_channels": 2,
        "is_segmentation": True,
        "layer_order": "gcr",
        "name": "ResidualUNet3D",
        "num_groups": 8}

    array = (1600 * scan + 400) / 1400
    array = np.clip(array, 0, 1)

    avg = patch[0] * patch[1] * patch[2] / (stride[0] * stride[1] * stride[2])
    slices = array_cut(scan, patch, stride)
    model = get_model(config_model)

    checkpoint = "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/artery_vein_stage_2.pytorch"
    model = load_checkpoint(checkpoint, model)
    model = model.to(device)
    model = model.to(torch.float16)

    array = sigmoid(array * 255, 20, 80)
    test_set = TestSetLoader(slices, array, device)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    prediction = predict_one_stage(test_loader, model, shape=array.shape, avg=avg, tissue="av")

    artery = np.array(prediction[0] > 0.52, "float32")
    vein = np.array(prediction[1] > 0.52, "float32")
    return artery, vein


def predict_airway(scan, device):
    patch = [512, 512, 32]
    stride = [512, 512, 8]

    avg = patch[0] * patch[1] * patch[2] / (stride[0] * stride[1] * stride[2])
    slices = array_cut(scan, patch, stride)
    test_set = TestSetLoader(slices, scan, device)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = U2NET(in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(
        "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/airway_segmentation.pth"))
    model = model.to(device)
    model = model.to(torch.float16)

    prediction = predict_one_stage(test_loader, model, shape=scan.shape, avg=avg, tissue="airway")
    airway = np.array(prediction[0] > 0.5, "float32")
    return airway


def predict_extra_av(scan, device):
    # model = UNet_av(in_channel=1, num_classes=3)
    # model.load_state_dict(torch.load(
    #     "/data/Train_and_Test/segmentation/chest_segmentation/predict_av_main.pth"))

    model = UNet_av(in_channel=1, num_classes=3)
    model.load_state_dict(torch.load(
        "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/predict_av_main_3.pth"))

    model = model.to(device)
    model = model.to(torch.float16)

    raw_array = preprocess(scan, device, tissue="av")
    with torch.no_grad():
        pre = model(raw_array).detach().cpu().numpy()
    # print(np.sum(pre[0, 0]), np.max(pre[0, 1]))
    pre_artery = np.array(pre[0, 0] > 0.52, "float32")
    pre_vein = np.array(pre[0, 1] > 0.52, "float32")
    return pre_artery, pre_vein


def predict_chest_segmentation(scan, do_filter=True, do_denoise=True, av_refine=True, visible_device=None, show=True):
    """
    :param show:
    :param visible_device: inference on single GPU. device should be like "0", "1"
    :param scan: the CT scan for predict. Signal normalization is [-600, 1000] -> [0, 1] rescaled Ct
    :param do_filter: do the second-stage av prediction with filtering. If not, do with DL method.
    :param do_denoise: whether do_denoise for the scan
    :param av_refine: whether twice_refine for artery-vein segmentation
    :return:
    """
    if visible_device is None:
        visible_device = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    if show:
        print("inference on GPU", visible_device, torch.cuda.get_device_name())

    device = torch.device("cuda")
    scan = np.clip(scan, -0.25, 1)

    if do_denoise:
        scan = predict_denoised_red(scan, device)

    lung = predict_lung_masks_rescaled_array(scan)

    loc = np.array(np.where(lung > 0))

    z_min, z_max = np.min(loc[2]), np.max(loc[2])
    valid_area = np.zeros(scan.shape, 'float32')
    valid_area[:, :, z_min:z_max] = 1
    scan *= valid_area

    array_airway = np.clip(scan, -0.25, 0.75) + 0.25
    airway = predict_airway(array_airway, device)

    array_av = np.clip(scan, -0.25, 1) + 0.25
    array_av = np.clip(rescale_to_new_shape(array_av, [256, 256, 256]), 0, 1)
    artery, vein = predict_extra_av(array_av, device)
    artery = rescale_to_new_shape(artery, [512, 512, 512])
    vein = rescale_to_new_shape(vein, [512, 512, 512])
    artery_1 = select_region(artery, num=2)
    vein_1 = select_region(vein, num=2)

    artery_1 = artery_1 * np.array(scan > 0, "float32")
    vein_1 = vein_1 * np.array(scan > 0, "float32")

    if do_filter:
        loc = np.array(np.where(artery_1 + vein_1 > 0))
        x_min, x_max = np.min(loc[0]), np.max(loc[0])
        y_min, y_max = np.min(loc[1]), np.max(loc[1])
        z_min, z_max = np.min(loc[2]), np.max(loc[2])

        filter_range = [max(x_min - 50, 0), min(x_max + 50, 512),
                        max(y_min - 50, 0), min(y_max + 50, 512),
                        max(z_min - 50, 0), min(z_max + 50, 512)]

        predict_filter = np.zeros(scan.shape)

        predict_filter[filter_range[0]:filter_range[1],
                       filter_range[2]:filter_range[3],
                       filter_range[4]:filter_range[5]] = \
            vessel_enhancement(scan[filter_range[0]:filter_range[1],
                               filter_range[2]:filter_range[3],
                               filter_range[4]:filter_range[5]],
                               lung[filter_range[0]:filter_range[1],
                               filter_range[2]:filter_range[3],
                               filter_range[4]:filter_range[5]])

        blood = np.array(predict_filter + artery_1 + vein_1 > 0.5, "float32")
        artery, vein = refinement(scan, artery_1, vein_1, blood, twice_refinement=av_refine, iteration=1)
    else:
        artery_2, vein_2 = predict_intra_av_2(scan, device)
        artery = select_region(np.array(artery_1 + artery_2 > 0.5, "float32"), num=2)
        vein = select_region(np.array(vein_1 + vein_2 > 0.5, "float32"), num=2)

    return lung, airway, artery, vein


if __name__ == '__main__':
    test_array = np.load('/data_disk/artery_vein_project/extract_blood_region/rescaled_ct-denoise/AL00029.npz')['array']

    lung_test, airway_test, artery_test, vein_test = predict_chest_segmentation(test_array, do_denoise=False)

    import visualization.visualize_3d.visualize_stl as stl
    stl.visualize_numpy_as_stl(lung_test)
    stl.visualize_numpy_as_stl(airway_test)
    stl.visualize_numpy_as_stl(artery_test)
    stl.visualize_numpy_as_stl(vein_test)
