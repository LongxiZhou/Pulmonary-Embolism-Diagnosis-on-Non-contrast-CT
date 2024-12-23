import numpy as np
import torch
import models.Unet_3D.U_net_Model_3D as U_net_Models
import os
import torch.nn as nn
import analysis.connect_region_detect as connected_region
import Tool_Functions.Functions as Functions
from format_convert.spatial_normalize import rescale_to_new_shape
import visualization.visualize_3d.visualize_stl as stl


def smooth_rescaled_mask(raw_vessel_mask_rescaled, get_connect=True, model=None, model_path=None, params=None,
                         show=True, new_array=True, normalize_to_256=False):
    """

    :param raw_vessel_mask_rescaled: artery mask or vein mask, in shape like [512, 512, 512], numpy float32

    :param normalize_to_256: if True, pad or down-sample the bounding box of blood vessel to (256, 256, 256)
    :param new_array: if False, will modify on the original array
    :param show:
    :param get_connect: if true, analysis connected component and return only one connected component
    :param model: loaded model on GPU
    :param model_path:
    :param params:
    :return: blood region, numpy array in float32, same shape with the input
    """
    rescaled_shape = np.shape(raw_vessel_mask_rescaled)
    assert len(rescaled_shape) == 3
    if get_connect:
        raw_vessel_mask_rescaled = connected_region.refine_connected_component(
            raw_vessel_mask_rescaled, 1, strict=False)

    bounding_box = Functions.get_bounding_box(raw_vessel_mask_rescaled)
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]
    if show:
        print("original bounding box shape:", (x_max - x_min, y_max - y_min, z_max - z_min))

    if normalize_to_256:
        if x_max - x_min < 256:
            x_min = max(0, int(x_min - (256 - (x_max - x_min)) / 2))
            x_max = x_min + 256
        if y_max - y_min < 256:
            y_min = max(0, int(y_min - (256 - (y_max - y_min)) / 2))
            y_max = y_min + 256
        if z_max - z_min < 256:
            z_min = max(0, int(z_min - (256 - (z_max - z_min)) / 2))
            z_max = z_min + 256
    else:

        pad_factor = 4  # model require shape to be 2^n, so pad to let shape to be 2^n
        assert rescaled_shape[0] % pad_factor == 0
        assert rescaled_shape[1] % pad_factor == 0
        assert rescaled_shape[2] % pad_factor == 0
        differ_x = pad_factor - (x_max - x_min) % pad_factor
        if differ_x < pad_factor:
            x_min = max(0, x_min - differ_x)
            if (x_max - x_min) % pad_factor > 0:
                x_max = x_max + pad_factor - (x_max - x_min) % pad_factor
        differ_y = pad_factor - (y_max - y_min) % pad_factor
        if differ_y < pad_factor:
            y_min = max(0, y_min - differ_y)
            if (y_max - y_min) % pad_factor > 0:
                y_max = y_max + pad_factor - (y_max - y_min) % pad_factor
        differ_z = pad_factor - (z_max - z_min) % pad_factor
        if differ_z < pad_factor:
            z_min = max(0, z_min - differ_z)
            if (z_max - z_min) % pad_factor > 0:
                z_max = z_max + pad_factor - (z_max - z_min) % pad_factor

    crop_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    if show:
        print("crop shape:", crop_shape, "inference shape:", (256, 256, 256) if normalize_to_256 else crop_shape)

    raw_mask = raw_vessel_mask_rescaled[x_min: x_max, y_min: y_max, z_min: z_max]
    if normalize_to_256:
        raw_mask = rescale_to_new_shape(raw_mask, target_shape=(256, 256, 256))
    blood_mask = inference_smooth_model_v1(raw_mask, model, model_path, params, False, show=show)
    if normalize_to_256:
        blood_mask_crop = rescale_to_new_shape(blood_mask, crop_shape)
        blood_mask_crop[blood_mask_crop > 0.5] = 1
    else:
        blood_mask_crop = blood_mask
    blood_mask_crop = connected_region.refine_connected_component(blood_mask_crop, 1, None, strict=False, show=show)
    if new_array:
        blood_region_rescaled = np.zeros(np.shape(raw_vessel_mask_rescaled), 'float32')
        blood_region_rescaled[x_min: x_max, y_min: y_max, z_min: z_max] = blood_mask_crop
    else:
        raw_vessel_mask_rescaled[x_min: x_max, y_min: y_max, z_min: z_max] = blood_mask_crop
        blood_region_rescaled = raw_vessel_mask_rescaled
    return blood_region_rescaled


def inference_smooth_model_v1(raw_vessel_mask, model=None, model_path=None, params=None, final_connect_refine=True,
                              show=True):
    """

    :param show:
    :param final_connect_refine:
    :param params:
    :param raw_vessel_mask: in shape [batch, x, y, z] or [x, y, z]
    :param model: loaded model on GPU
    :param model_path:
    :return: blood region, numpy array in float32, same shape with the input
    """
    if params is None:
        params = {"model_size": 'small', "in_channels": 1, "out_channels": 2, "init_features": 16,
                  "device": "cuda:0" if torch.cuda.is_available() else "cpu"}

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    shape = np.shape(raw_vessel_mask)
    stack_in_batch = False
    if len(shape) == 4:
        stack_in_batch = True
    else:
        assert len(shape) == 3

    if stack_in_batch is False:
        raw_vessel_mask = np.reshape(raw_vessel_mask, (1, 1, shape[0], shape[1], shape[2]))
    else:
        raw_vessel_mask = np.reshape(raw_vessel_mask, (shape[0], 1, shape[1], shape[2], shape[3]))

    if model is None:
        if model_path is None:
            model_path = '/data_disk/artery_vein_project/smooth_blood_mask/check_points/256_final/' \
                         'best_model_smooth.pth'
        if params["model_size"] == 'large':
            model = U_net_Models.UNet3D(params["in_channels"], params["out_channels"], params["init_features"])
        elif params["model_size"] == 'median':
            model = U_net_Models.UNet3DSimple(params["in_channels"], params["out_channels"], params["init_features"])
        else:
            assert params["model_size"] == 'small'
            model = U_net_Models.UNet3DSimplest(params["in_channels"], params["out_channels"], params["init_features"])
        model = model.to(params["device"])

        if torch.cuda.device_count() > 1 and stack_in_batch and shape[0] > 1:
            model = nn.DataParallel(model)
        data_dict = torch.load(model_path)
        if type(model) == nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])

    softmax_layer = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(raw_vessel_mask).cuda()
        print(input_tensor.shape)
        segmentation_before_softmax = model(input_tensor)
        segment_probability_lesion = softmax_layer(segmentation_before_softmax).cpu().numpy()[:, 0]  # [B, x, y, z]
    segment_mask_lesion = np.array(segment_probability_lesion > 0.5, 'float32')
    blood_region = raw_vessel_mask[:, 0] - segment_mask_lesion  # [B, x, y, z]
    if final_connect_refine:
        for sample in range(np.shape(blood_region)[0]):
            blood_region[sample] = connected_region.refine_connected_component(blood_region[sample], 1, strict=False,
                                                                               show=show)

    if stack_in_batch:
        return blood_region
    return blood_region[0]


if __name__ == '__main__':

    # test_array = np.load(
    #     '/data_disk/artery_vein_project/temp_data/CS1_artery.npz')['arr_0']

    test_array = np.load('/data_disk/artery_vein_project/smooth_blood_mask/training_data/'
                         'sliced_sample/256/non-contrast/stack_array_vein/PL00002.npz')['array'][0]

    print(np.shape(test_array))
    refined_array = smooth_rescaled_mask(test_array, normalize_to_256=True)
    stl.visualize_numpy_as_stl(test_array)
    stl.visualize_numpy_as_stl(refined_array)

    exit()
    test_array = np.load('/data_disk/artery_vein_project/extract_blood_region/training_data/'
                         'sliced_sample/256/non-contrast/stack_array_artery/PL00032.npz')['array'][0]

    refined_array = inference_smooth_model_v1(test_array)
    stl.visualize_numpy_as_stl(test_array)
    stl.visualize_numpy_as_stl(refined_array)
