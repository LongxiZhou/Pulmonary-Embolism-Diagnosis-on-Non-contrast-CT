"""
input: ct signal rescaled (lung window to -0.5, 0.5) and spatial rescaled in [x, y], while z thickness is 5 mm.
output: rescaled ct
"""

import format_convert.dcm_np_converter as converter
import format_convert.spatial_normalize as spatial_normalize
import basic_tissue_prediction.predict_rescaled as predictor
import Tool_Functions.Functions as Functions
import models.Unet_2D.test as test_model
import numpy as np
import warnings


def up_sample_5mm(rescaled_array, check_point_path, batch_size=16):

    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 5,  # positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 64,
        "encoder_blocks": 2
    }
    print("up-sampling z...\n")

    up_sampled_array = test_model.predict_one_scan_multi_class(rescaled_array, 'Z', check_point_path,
                                                               array_info, batch_size=batch_size, soft_max=False)
    return up_sampled_array  # (x, y, z, 5)


def from_5mm_dcm_to_rescaled(dict_dcm, check_point_path, batch_size=16):
    stack_simple, resolution = converter.dcm_to_unrescaled(dict_dcm, return_resolution=True)
    stack_simple = (stack_simple + 600) / 1600
    resolution = list(resolution)
    shape_original = np.shape(stack_simple)

    if not resolution[2] == 5:
        warnings.warn("slice thickness should be 5mm, change to 5mm")
        resolution[2] = 5
    if not np.shape(stack_simple[:, :, 0]) == (512, 512):
        warnings.warn("unexpected shape in x-y")
    temp_stack_array = np.zeros([512, 512, shape_original[2] * 5], 'float32')

    up_sample_array = up_sample_5mm(stack_simple, check_point_path, batch_size)

    for z in range(shape_original[2]):
        temp_stack_array[:, :, z * 5] = up_sample_array[:, :, z, 0]
        temp_stack_array[:, :, z * 5 + 1] = up_sample_array[:, :, z, 1]
        temp_stack_array[:, :, z * 5 + 2] = up_sample_array[:, :, z, 2]
        temp_stack_array[:, :, z * 5 + 3] = up_sample_array[:, :, z, 3]
        temp_stack_array[:, :, z * 5 + 4] = up_sample_array[:, :, z, 4]

    rescaled_ct = spatial_normalize.rescale_to_standard(temp_stack_array, (resolution[0], resolution[1], 1))

    return rescaled_ct


def upsample_array_z_from_5mm_to_1mm(signal_rescaled_ct, check_point_path=None, batch_size=16):
    """

    :param signal_rescaled_ct: numpy array in shape [512, 512, z], lung window cast to [-0.5, 0.5],
    the resolution on z axis must be 5mm.
    :param check_point_path: check_point_path for the upsample model_guided
    :param batch_size:
    :return: high resolution ct in shape [512, 512, z * 5]
    """
    shape_input_ct = np.shape(signal_rescaled_ct)
    assert shape_input_ct[0] == 512 and shape_input_ct[1] == 512
    if check_point_path is None:
        check_point_path = \
            '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/check_point_refine_L2F64/0_saved_model.pth'
    upsampled_array = np.zeros([512, 512, shape_input_ct[2] * 5], 'float32')

    rescaled_ct = up_sample_5mm(signal_rescaled_ct, check_point_path, batch_size)

    for z in range(shape_input_ct[2]):
        upsampled_array[:, :, z * 5] = rescaled_ct[:, :, z, 0]
        upsampled_array[:, :, z * 5 + 1] = rescaled_ct[:, :, z, 1]
        upsampled_array[:, :, z * 5 + 2] = rescaled_ct[:, :, z, 2]
        upsampled_array[:, :, z * 5 + 3] = rescaled_ct[:, :, z, 3]
        upsampled_array[:, :, z * 5 + 4] = rescaled_ct[:, :, z, 4]

    return upsampled_array


if __name__ == '__main__':
    import visualization.visualize_3d.visualize_stl as stl
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'

    ct_up_sampled = from_5mm_dcm_to_rescaled('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/good_CTA-CT_interval_but_bad_dcm/N324/non-contrast/',
                                             '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/check_point_refine_L2F64/0_saved_model.pth', batch_size=32)

    exit()
    ct_up_sampled_2 = from_5mm_dcm_to_rescaled('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000022/5mm/',
                                             '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/check_point_refine_L2F64/0_saved_model.pth', batch_size=1)
    print(np.sum(np.abs((ct_up_sampled_2 - ct_up_sampled))))
    exit()

    lung = predictor.predict_lung_masks_rescaled_array(ct_up_sampled)
    stl.visualize_numpy_as_stl(lung)
    blood_vessel = predictor.get_prediction_blood_vessel(ct_up_sampled, lung_mask=lung)
    stl.visualize_numpy_as_stl(blood_vessel)

    for z in range(150, 350, 10):
        Functions.image_show(np.clip(ct_up_sampled[:, :, z], -0.5, 0.5), gray=True)
    exit()
    exit()

    original_rescaled = converter.dcm_to_signal_rescaled('/home/zhoul0a/Desktop/pulmonary_embolism/dcm_and_gt/1/non-contrast/', wc_ww=(-600, 1600))
    original_rescaled = spatial_normalize.rescale_to_standard(original_rescaled, [0.755859375, 0.755859375, 5])
    Functions.image_show(original_rescaled[:, :, 256])
    lung = predictor.predict_lung_masks_rescaled_array(original_rescaled)
    Functions.merge_image_with_mask(np.clip(original_rescaled[:, :, 256] + 0.5, 0, 1), lung[:, :, 256])
    stl.visualize_numpy_as_stl(lung)
    blood_vessel = predictor.get_prediction_blood_vessel(original_rescaled, lung_mask=lung)
    stl.visualize_numpy_as_stl(blood_vessel)
    exit()


