"""
We will predict a artery of expected values, and compare it with the ground truth value (rescaled_ct).
To remove blood wall, we will remove voxels with encoding_depth less than 4.
The encoding_depth will be normalized to max 27. The median for encoding_depth is 27, std for encoding_depth is 1.78.
The following are the radiomics:

1) L1_distance:
voxel_wise average of the ground truth higher than expected values.

2) depth_weighted_L1_distance:
voxel_wise average of the ground truth higher than expected values, and the weight for each voxel is related to the
encoding_depth.

3) L2_distance:
voxel_wise average of the square difference of ground truth and expected values.

4) depth_weighted_L2_distance:
voxel_wise average of the square difference of ground truth and expected values, and the weight for each voxel is
related to the encoding_depth.

5) center_line_L1_distance:
voxel_wise average of the ground truth higher than expected values, but on return_center_line only.

6) depth_weighted_center_line_L1_distance:
similar with "center_line_L1_distance", but each voxel is added the encoding_depth weight.

7) center_line_L2_distance:
voxel_wise average of L2 difference of ground truth and expected values, but on return_center_line only.

8) depth_weighted_center_line_L2_distance:
similar with "center_line_L2_distance", but each voxel is added the encoding_depth weight.

9) Anderson-Darling test statistic for the difference of prediction and expected
The Anderson-Darling test is a normality test based on the accumulative difference between empirical cumulative
distribution function and the expected cumulative distribution function.

10) Anderson-Darling test statistic for the difference of prediction and expected, on center line only.
"""

import numpy as np
import Tool_Functions.Functions as Functions
import os
import warnings
import analysis.center_line_and_depth_3D as center_line_and_distance
import med_transformer.image_transformer.transformer_for_3D.rescaled_ct_sample_sequence_converter as reconstruct
import format_convert.spatial_normalize as spatial_normalize


def get_depth_weighted_l1_distance(rescaled_ct, predicted_ct, blood_vessel_mask, depth_mask=None, ):
    """

    :param rescaled_ct: rescaled_ct in shape [512, 512, 512]
    :param predicted_ct: in shape [512, 512, 512], only need to predict the blood vessel regions.
    :param blood_vessel_mask: binary mask in shape [512, 512, 512]
    :param depth_mask: the encoding_depth of voxel to the blood vessel surface.
    :return: the radiomic
    """
    pass


def get_inherent_noise(rescaled_ct, blood_vessel_depth, is_depth_mask=False, show=True, depth_difference=5,
                       denoise=True):
    """

    :param rescaled_ct:
    :param blood_vessel_depth: can provide blood vessel mask or blood vessel encoding_depth mask
    :param is_depth_mask:
    :param show
    :param depth_difference
    :param denoise: whether the rescaled_ct is denoise
    :return: case dict for inherent noise
    """
    if not is_depth_mask:
        blood_vessel_depth = center_line_and_distance.get_surface_distance(blood_vessel_depth)

    max_depth = np.max(blood_vessel_depth)
    assert max_depth > depth_difference > 0

    if max_depth < 20:
        print("max encoding_depth is:", max_depth)
        warnings.warn("Value Warning: the max encoding_depth of the blood vessel is too small.")

    if show:
        print("max encoding_depth is:", max_depth)

    mask_sampling = np.array(blood_vessel_depth > (max_depth - depth_difference), 'int16')
    non_zero_loc = Functions.get_location_list(np.where(mask_sampling))

    ct_value = []
    for loc in non_zero_loc:
        ct_value.append(rescaled_ct[loc])
    ct_value = np.array(ct_value)
    ct_value = ct_value * 1600 - 600

    mean = np.mean(ct_value)
    std = np.std(ct_value)
    abs_differ_ave = np.mean(np.abs(ct_value - mean))

    if show:
        print("total sampled:", len(ct_value))
        print('mean:', mean, 'std:', std)
        print('abs differ ave:', abs_differ_ave)

        print("mean value 95% sample_interval: [", mean - 2.3 * std / np.sqrt(len(ct_value)), ',',
              mean + 2.3 * std / np.sqrt(len(ct_value)), ']')

    if denoise:
        case_dict = {"inherent_noise (denoise)": abs_differ_ave, "noise_std (denoise)": std,
                     "sampling_points_for_noise (denoise)": len(ct_value), "average_blood HU (denoise)": mean}
    else:
        case_dict = {"inherent_noise": abs_differ_ave, "noise_std": std,
                     "sampling_points_for_noise": len(ct_value), "average_blood HU": mean, "max_blood_depth": max_depth}

    if show:
        print(case_dict)

    return case_dict


def get_inherent_noise_from_sequence(sample_sequence, absolute_cube_length, min_depth=15):
    rescaled_ct = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(
        sample_sequence, absolute_cube_length, key='ct_data')
    depth_mask = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(
        sample_sequence, absolute_cube_length, key='depth_cube')

    rescaled_ct = spatial_normalize.rescale_to_new_shape(rescaled_ct, [233, 233, 233])
    depth_mask = spatial_normalize.rescale_to_new_shape(depth_mask, [233, 233, 233])

    mask_greater_15 = np.array(depth_mask > min_depth, 'float32')

    non_zero_loc = Functions.get_location_list(np.where(mask_greater_15))

    rescaled_ct = rescaled_ct * mask_greater_15

    ct_value = []

    for loc in non_zero_loc:
        ct_value.append(rescaled_ct[loc])

    ct_value = np.array(ct_value)

    ct_value = ct_value * 1600 - 600

    mean = np.mean(ct_value)
    std = np.std(ct_value)
    abs_differ_ave = np.mean(np.abs(ct_value - mean))
    print("total sampled:", len(ct_value))
    print('mean:', mean, 'std:', std)
    print('abs differ ave:', abs_differ_ave)

    print("mean value 95% sample_interval: [", mean - 2.3 * std / np.sqrt(len(ct_value)), ',',
          mean + 2.3 * std / np.sqrt(len(ct_value)), ']')

    return abs_differ_ave


def get_all_inherent_noise(absolute_cube_length):
    top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset/merged_v1/'

    fn_list = os.listdir(top_dict)

    if os.path.exists('/home/zhoul0a/Desktop/pulmonary_embolism/analysis/report_for_inherent_noise.pickle'):
        report = Functions.pickle_load_object(
            '/home/zhoul0a/Desktop/pulmonary_embolism/analysis/report_for_inherent_noise.pickle')
    else:
        report = {}  # (sequence_name: (max_depth, number voxel > 15, mean for encoding_depth > 15 (HU), std, abs differ ave))

    processed = list(report.keys())

    def get_inherent_noise_scan(fn):
        print(fn)
        if fn in processed:
            print("processed")
            return None

        sample_sequence = Functions.pickle_load_object(top_dict + fn)

        rescaled_ct = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(
            sample_sequence, absolute_cube_length, key='ct_data')

        depth_mask = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(
            sample_sequence, absolute_cube_length, key='depth_cube')

        rescaled_ct = spatial_normalize.rescale_to_new_shape(rescaled_ct, [233, 233, 233])
        depth_mask = spatial_normalize.rescale_to_new_shape(depth_mask, [233, 233, 233])

        mask_greater_15 = np.array(depth_mask > 15, 'float32')

        non_zero_loc = Functions.get_location_list(np.where(mask_greater_15))

        rescaled_ct = rescaled_ct * mask_greater_15

        ct_value = []

        for loc in non_zero_loc:
            ct_value.append(rescaled_ct[loc])

        ct_value = np.array(ct_value)

        ct_value = ct_value * 1600 - 600

        mean = np.mean(ct_value)
        std = np.std(ct_value)
        abs_differ_ave = np.mean(np.abs(ct_value - mean))
        print('mean:', mean, 'std:', std)
        print('abs differ ave:', abs_differ_ave)

        report[fn] = (np.max(depth_mask), np.sum(mask_greater_15), mean, std, abs_differ_ave)

        Functions.pickle_save_object(
            '/home/zhoul0a/Desktop/pulmonary_embolism/analysis/report_for_inherent_noise.pickle',
            report)
        processed.append(fn)

    for sample_name in fn_list:
        get_inherent_noise_scan(sample_name)
        print(len(fn_list) - len(processed), 'left')


if __name__ == '__main__':
    exit()
