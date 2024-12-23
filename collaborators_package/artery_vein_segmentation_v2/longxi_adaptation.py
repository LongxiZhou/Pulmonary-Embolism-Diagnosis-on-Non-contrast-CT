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
from collaborators_package.do_filter.frangi_3d_parallel import vessel_enhancement_parallel
from visualization.visualize_3d import visualize_stl as view
from collaborators_package.artery_vein_segmentation_v2.artery_vein.artery_vein_refine import refinement
from basic_tissue_prediction.predict_rescaled import predict_lung_masks_rescaled_array


def pre_process(pre_array, device, tissue="airway"):
    array = pre_array[np.newaxis]
    if tissue == "av":
        array = array[np.newaxis]
    array = torch.HalfTensor(array)
    array = array.to(device)
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
        return area, pre_process(sub_array, self.device)

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
        prediction = np.zeros([1, shape[0], shape[1], shape[2]], "float32")
    elif tissue == "av":
        prediction = np.zeros([2, shape[0], shape[1], shape[2]], "float32")
    else:
        return None

    for iteration, (area, sub_array) in enumerate(test_loader):
        if torch.sum(sub_array) == 0:
            # print("pass")
            continue
        with torch.no_grad():
            predict_result = model(sub_array)[0].detach().cpu().numpy()
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

    # array = (1600 * scan + 400) / 1400
    array = 8 / 7 * scan + 2 / 7
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


def load_airway_model(path=None, device=None):
    if path is None:
        path = "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/airway_segmentation.pth"
    if device is None:
        device = torch.device("cuda:0")
    model = U2NET(in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model = model.to(torch.float16)

    return model


def predict_airway(scan, device=None, to_binary=True, model_loaded=None):
    patch = [512, 512, 32]
    stride = [512, 512, 8]

    avg = patch[0] * patch[1] * patch[2] / (stride[0] * stride[1] * stride[2])
    slices = array_cut(scan, patch, stride)
    test_set = TestSetLoader(slices, scan, device)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    if model_loaded is None:
        model = load_airway_model(None, device=device)
    else:
        model = model_loaded

    prediction = predict_one_stage(test_loader, model, shape=scan.shape, avg=avg, tissue="airway")
    if to_binary:
        airway = np.array(prediction[0] > 0.5, "float32")
    else:
        airway = np.array(prediction[0], 'float32')
    return airway


def load_av_model(path=None, device=None):
    if path is None:
        # path = "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/predict_av_main_3.pth"
        path = "/home/zhoul0a/Desktop/prognosis_project/check_points/chest_segmentation/predict_av_main_3_unzip.pth"
    if device is None:
        device = torch.device("cuda:0")
    model = UNet_av(in_channel=1, num_classes=3)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model = model.to(torch.float16)

    return model


def predict_extra_av(scan, device=None, to_binary=True, model_loaded=None):
    if model_loaded is None:
        model = load_av_model(None, device=device)
    else:
        model = model_loaded

    raw_array = pre_process(scan, device, tissue="av")
    with torch.no_grad():
        pre = model(raw_array).detach().cpu().numpy()

    if to_binary:
        pre_artery = np.array(pre[0, 0] > 0.5, "float32")
        pre_vein = np.array(pre[0, 1] > 0.5, "float32")
    else:
        pre_artery = np.array(pre[0, 0], "float32")
        pre_vein = np.array(pre[0, 1], "float32")
    return pre_artery, pre_vein


def predict_airway_rescaled(rescaled_ct, loaded_model=None, visible_device=None, clip=True, to_binary=True,
                            bounding_box=None):
    """

    :param bounding_box: like ((100, 300), (150, 350), (50, 400)), semantic only exists in this bounding box,
    note different model requires different bounding box, like CNN requires shape for each dim % 2^num_max_pool == 0
    :param loaded_model:
    :param to_binary: whether output binary mask or probability map
    :param clip: some scan may have HU from -3000 - 3000, clip it to -1000, 1000
    :param rescaled_ct:
    :param visible_device: inference on single GPU. device should be like "0", "1"
    :return: airway mask/probability in shape same with rescaled_ct
    """
    if visible_device is not None:
        device = torch.device("cuda:" + visible_device)
    else:
        device = torch.device("cuda")
    if clip:
        rescaled_ct = np.clip(rescaled_ct, -0.25, 1)
    if bounding_box is None:
        return predict_airway(scan=rescaled_ct, device=device, to_binary=to_binary, model_loaded=loaded_model)
    assert len(bounding_box) == 3
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]
    airway_array = np.zeros(np.shape(rescaled_ct), 'float32')
    rescaled_ct_bounded = rescaled_ct[x_min: x_max, y_min: y_max, z_min: z_max]
    airway_array[x_min: x_max, y_min: y_max, z_min: z_max] = \
        predict_airway(scan=rescaled_ct_bounded, device=device, to_binary=to_binary, model_loaded=loaded_model)
    return airway_array


def predict_av_rescaled(rescaled_ct, model_path=None, loaded_model=None, visible_device=None, clip=True,
                        bounding_box=None, lung_mask=None, refine=False, show=True, max_parallel=48):
    """

    :param model_path:
    :param max_parallel:
    :param show:
    :param refine:
    :param lung_mask: if not None, use it to calculate bounding_box
    :param bounding_box: like ((100, 300), (150, 350), (50, 400)), semantic only exists in this bounding box,
    note different model requires different bounding box, like CNN requires shape for each dim % 2^num_max_pool == 0
    :param loaded_model:
    :param clip: some scan may have HU from -3000 - 3000, clip it to -1000, 1000
    :param rescaled_ct:
    :param visible_device: inference on single GPU. device should be like "0", "1"
    :return: airway mask/probability in shape same with rescaled_ct
    """
    if visible_device is not None:
        device = torch.device("cuda:" + visible_device)
    else:
        device = torch.device("cuda")
    if clip:
        rescaled_ct = np.clip(rescaled_ct, -0.25, 1)

    rescaled_shape = np.shape(rescaled_ct)

    pad_bounding_box = 4
    if bounding_box is not None:
        x_min, x_max = bounding_box[0]
        y_min, y_max = bounding_box[1]
        z_min, z_max = bounding_box[2]
    else:
        if lung_mask is None:
            lung_mask = predict_lung_masks_rescaled_array(rescaled_ct)
        loc = np.array(np.where(lung_mask > 0))
        x_min, x_max = np.min(loc[0]), np.max(loc[0])
        y_min, y_max = np.min(loc[1]), np.max(loc[1])
        z_min, z_max = np.min(loc[2]), np.max(loc[2])
    x_min, x_max = max(0, x_min - pad_bounding_box), min(rescaled_shape[0], x_max + pad_bounding_box)
    y_min, y_max = max(0, y_min - pad_bounding_box), min(rescaled_shape[1], y_max + pad_bounding_box)
    z_min, z_max = max(0, z_min - pad_bounding_box), min(rescaled_shape[2], z_max + pad_bounding_box)

    crop_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    if show:
        print("crop shape:", crop_shape)

    final_artery_mask = np.zeros(rescaled_shape, 'float32')
    final_vein_mask = np.zeros(rescaled_shape, 'float32')

    rescaled_clip = rescaled_ct[x_min: x_max, y_min: y_max, z_min: z_max]

    if show:
        print("predicting root")
    array_av = rescaled_ct + 0.25
    array_av = np.clip(rescale_to_new_shape(array_av, [256, 256, 256]), 0, 1)
    if model_path is not None:
        assert loaded_model is None
        loaded_model = load_av_model(model_path, device=device)
    artery, vein = predict_extra_av(array_av, device, model_loaded=loaded_model)
    artery = select_region(artery, 1)
    vein = select_region(vein, 1)
    artery = rescale_to_new_shape(artery, [512, 512, 512])
    vein = rescale_to_new_shape(vein, [512, 512, 512])

    if show:
        print("predicting small vessel")
    valid_mask_clip = np.array(rescaled_clip > 0.2)
    predict_filter = vessel_enhancement_parallel(
        rescaled_clip, lung_mask[x_min: x_max, y_min: y_max, z_min: z_max], num_workers=max_parallel)
    artery_clip = artery[x_min: x_max, y_min: y_max, z_min: z_max]
    vein_clip = vein[x_min: x_max, y_min: y_max, z_min: z_max]
    blood_clip = np.array(predict_filter + artery_clip + vein_clip > 0.5, "float32")
    blood_clip = blood_clip * valid_mask_clip
    blood_clip = select_region(blood_clip, 2)

    if show:
        print("propagate semantic and refine")
    artery_clip, vein_clip = refinement(rescaled_clip, artery_clip, vein_clip, blood_clip,
                                        twice_refinement=refine, iteration=1, max_parallel_count=max_parallel)

    artery_clip = select_region(artery_clip, 1)
    vein_clip = select_region(vein_clip, 1)

    final_artery_mask[x_min: x_max, y_min: y_max, z_min: z_max] = artery_clip
    final_vein_mask[x_min: x_max, y_min: y_max, z_min: z_max] = vein_clip

    return final_artery_mask, final_vein_mask


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
    scan[:, :, 0: z_min] = 0
    scan[:, :, z_max::] = 0

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
    import visualization.visualize_3d.visualize_stl as stl
    test_array = np.load('/data_disk/artery_vein_project/extract_blood_region/rescaled_ct-denoise/AL00029.npz')['array']
    test_lung = np.load('/data_disk/artery_vein_project/extract_blood_region/semantics/lung_mask/AL00029.npz')['array']
    test_artery, test_vein = predict_av_rescaled(test_array, visible_device=None, lung_mask=test_lung, refine=True)
    stl.visualize_numpy_as_stl(test_artery)
    stl.visualize_numpy_as_stl(test_vein)
    exit()
    test_airway = predict_airway_rescaled(test_array, visible_device='1')
    stl.visualize_numpy_as_stl(test_airway)
    exit()

    lung_test, airway_test, artery_test, vein_test = predict_chest_segmentation(test_array, do_denoise=False)
    stl.visualize_numpy_as_stl(lung_test)
    stl.visualize_numpy_as_stl(airway_test)
    stl.visualize_numpy_as_stl(artery_test)
    stl.visualize_numpy_as_stl(vein_test)
