"""
Vessels in CTA has high average HU value, usually higher than 200, which is significantly higher than adjacent tissue.

source file from:
/data_disk/artery_vein_project/extract_blood_region/paired_vessel_predicted_and_gt

each file is a binary numpy array in [6, 512, 512, 512], channel is:
0 artery_blood_region: HU > 150 in CTA, no a-v overlap
1 artery_annotation: the artery ground truth provided by Prof Qiu, but no a-v overlap
2 artery_prediction: convert CTA to CT, and run a-v seg
3 vein_blood_region: HU > 150 in CTA, no a-v overlap
4 vein_annotation: the artery ground truth provided by Prof Qiu, but no a-v overlap
5 vein_prediction: convert CTA to CT, and run a-v seg

Three types of sample:
A: input a-v annotation, output blood region
B: input a-v prediction, output blood region, i.e., more difficult than A
C: input a-v prediction plus lesions merged with a-v like nodule, i.e., the most difficult task

All sample types shaped [2, 64, 64, 64], each channel is:
channel 0: input, artery_annotation 1 and vein_annotation -1, other 0
channel 1: ground truth for blood region, artery_blood_region 1 and vein_blood_region -1, other 0

Model only do reduction: reduce the input high recall mask to get the blood region

"""
import numpy as np
import Tool_Functions.Functions as Functions
import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.center_line_and_depth_3D as get_center_line
import random
import os
from format_convert.spatial_normalize import rescale_to_new_shape


def get_blood_region(rescaled_ct_denoise, raw_mask, case_report=None):
    """

    this seems not a good way for regular CTA. As the signal in blood vessels varies very significantly:
    sometimes vein has signal hundreds HU higher than artery, vice versa.

    :param rescaled_ct_denoise: CTA
    :param raw_mask: the mask of raw blood mask
    :param case_report:
    :return: the blood region of the
    """
    if case_report is not None:
        key_list = list(case_report.keys())
        if 'average_blood HU (denoise)' in key_list:
            assert case_report['average_blood HU (denoise)'] >= 250
        else:
            assert case_report['average_blood HU'] >= 250

    blood_region = np.array(rescaled_ct_denoise * raw_mask > Functions.change_to_rescaled(100), 'float32')
    return blood_region


def get_artery_vein_blood_region(rescaled_ct_denoise, raw_mask_artery, raw_mask_vein, case_report=None):
    """

    :param rescaled_ct_denoise: CTA
    :param raw_mask_artery:
    :param raw_mask_vein:
    :param case_report:
    :return: blood_region_artery, blood_region_vein
    """
    if case_report is not None:
        key_list = list(case_report.keys())
        if 'average_blood HU (denoise)' in key_list:
            assert case_report['average_blood HU (denoise)'] >= 250
        else:
            assert case_report['average_blood HU'] >= 250

    blood_region_artery = get_blood_region(rescaled_ct_denoise, raw_mask_artery, None)
    blood_region_vein = get_blood_region(rescaled_ct_denoise, raw_mask_vein, None)
    overlap = blood_region_artery * blood_region_vein
    blood_region_artery = blood_region_artery - overlap
    blood_region_vein = blood_region_vein - overlap

    return blood_region_artery, blood_region_vein


def get_input_output_and_penalty_array(blood_vessel_mask=None, blood_center_line_mask=None):
    """
    total penalty for fn is 100 * sqrt(num_gt_output_voxel)

    model input: blood_vessel_mask + lesion (optional)
    model output: region to reduce (blood region = model_input - model_out)

    :param blood_center_line_mask: blood_center_line for the blood region
    :param blood_vessel_mask
    :return: raw_mask (input for model), raw_mask - blood_region (output for model), penalty for false negative,
    blood region, penalty for false positive
    """
    raw_mask = np.array(blood_vessel_mask, 'float32')

    # apply_lesion will change model_output_gt and penalty_fn, initially it is zeros
    model_output_gt = np.zeros(np.shape(raw_mask), 'float32')
    penalty_fn = np.zeros(np.shape(raw_mask), 'float32')

    # set background average_penalty_fn to 1/3
    average_penalty_fn = 1 / 3

    # in this task blood region is same with raw_mask
    blood_region = raw_mask
    return_list = [raw_mask, model_output_gt, penalty_fn, blood_region]

    # The following calculates the penalty for fp
    # model only do reduction, high penalty is given for background regions
    penalty_fp = average_penalty_fn - average_penalty_fn * raw_mask
    total_penalty = average_penalty_fn * np.sum(blood_region)
    # apply total_penalty to blood_region
    penalty_fp = penalty_fp + blood_region * (total_penalty / np.sum(blood_region))
    # furthermore, for deeper region we give high penalty
    # center line give very high penalty
    if blood_center_line_mask is None:
        blood_center_line_mask = get_center_line.get_center_line(blood_region) * blood_region
    penalty_fp = penalty_fp + blood_region * blood_center_line_mask * (
            total_penalty / np.sum(blood_center_line_mask))
    # surface give high penalty
    surface_layer_1 = get_surface.get_surface(blood_region, outer=False, strict=False)
    penalty_fp = penalty_fp + surface_layer_1 * (total_penalty / np.sum(surface_layer_1))

    return_list.append(penalty_fp)

    return return_list


def apply_lesion_parallel(*args):
    """
    :return: lesion_loc_aligned

    raw_mask_with_lesion: raw_mask[lesion_loc_aligned] = 1
    model_output_gt (with lesion): model_output_gt[lesion_loc_aligned] = 1
    penalty_fn (with_lesion): penalty_fn[lesion_loc_aligned] = lesion_penalty
    """
    assert len(args) == 2
    send_end = args[0]  # send end for multiprocessing.Pipe(False)
    input_list = args[1]  # each item is raw_mask, model_output_gt, list_lesion_loc_array

    if input_list is None:
        send_end.send(None)
        send_end.close()
        return None

    collection_list = []  # collect results for this thread
    for raw_mask, model_output_gt, list_lesion_loc_array in input_list:
        if list_lesion_loc_array is None:
            collection_list.append(None)
            continue

        none_count = 0
        for lesion_loc_array in list_lesion_loc_array:
            if lesion_loc_array is None:
                none_count += 1
        if none_count == len(list_lesion_loc_array):
            collection_list.append(None)
            continue

        loc_array_input = np.where(raw_mask > 0)
        if len(loc_array_input[0]) == 0:
            collection_list.append(None)
            continue

        raw_mask_with_lesion = np.array(raw_mask, 'float32')
        shape_mask = np.shape(raw_mask_with_lesion)

        volume_array_temp = np.zeros(np.shape(raw_mask), 'float32')

        for lesion_loc_array in list_lesion_loc_array:
            if lesion_loc_array is None:
                continue
            origin_lesion_mass_center = [int(np.median(lesion_loc_array[0])), int(np.median(lesion_loc_array[1])),
                                         int(np.median(lesion_loc_array[2]))]

            pixel_id = random.randint(0, len(loc_array_input[0]) - 1)
            # pick a random location
            center_loc_lesion = (loc_array_input[0][pixel_id], loc_array_input[1][pixel_id],
                                 loc_array_input[2][pixel_id])
            lesion_loc_aligned = (lesion_loc_array[0] - origin_lesion_mass_center[0] + center_loc_lesion[0],
                                  lesion_loc_array[1] - origin_lesion_mass_center[1] + center_loc_lesion[1],
                                  lesion_loc_array[2] - origin_lesion_mass_center[2] + center_loc_lesion[2])

            # trim the lesion loc
            lesion_loc_aligned = (np.clip(np.array(lesion_loc_aligned[0], 'int32'), 0, shape_mask[0] - 1),
                                  np.clip(np.array(lesion_loc_aligned[1], 'int32'), 0, shape_mask[1] - 1),
                                  np.clip(np.array(lesion_loc_aligned[2], 'int32'), 0, shape_mask[2] - 1))

            raw_mask_with_lesion[lesion_loc_aligned] = 1
            volume_array_temp[lesion_loc_aligned] = len(lesion_loc_aligned[0])

        update_lesion_array = raw_mask_with_lesion - raw_mask
        lesion_loc_merged = np.where(update_lesion_array > 0.5)
        volume_list = volume_array_temp[lesion_loc_merged]
        lesion_volume_array = np.array(volume_list, 'float32')
        collection_list.append((lesion_loc_merged, lesion_volume_array))

    send_end.send(collection_list)
    send_end.close()
    return collection_list


def get_stack_array_256(semantic='artery', fold=(0, 1), cta=False, correct_annotation_example_dir=None):
    """

        form the stack array for each case (no lesion):
        array with shape [4, 256, 256, 256], each channel is: raw_mask, model_output_gt, penalty_fn, penalty fp

        :param correct_annotation_example_dir:
        :param cta:
        :param fold:
        :param semantic:
        :return:
        """
    if cta:
        data_top_dict = '/data_disk/artery_vein_project/extract_blood_region/'
        save_top_dict = '/data_disk/artery_vein_project/smooth_blood_mask/training_data/sliced_sample/256/CTA/'
        if correct_annotation_example_dir is not None:
            correct_annotation_example_dir = os.path.join(correct_annotation_example_dir, 'CTA')
    else:
        data_top_dict = '/data_disk/artery_vein_project/new_data/non-contrast/'
        save_top_dict = \
            '/data_disk/artery_vein_project/smooth_blood_mask/training_data/sliced_sample/256/non-contrast/'
        if correct_annotation_example_dir is not None:
            correct_annotation_example_dir = os.path.join(correct_annotation_example_dir, 'non-contrast')

    assert semantic in ['artery', 'vein']
    correct_fn_list = None
    if semantic == 'artery':
        save_dict = save_top_dict + 'stack_array_artery/'
        vessel_mask_dict = data_top_dict + 'semantics/artery_mask/'
        if correct_annotation_example_dir is not None:
            correct_example_fn_dir = os.path.join(correct_annotation_example_dir, 'stack_array_artery')
            correct_fn_list = os.listdir(correct_example_fn_dir)
    else:
        save_dict = save_top_dict + 'stack_array_vein/'
        vessel_mask_dict = data_top_dict + 'semantics/vein_mask/'
        if correct_annotation_example_dir is not None:
            correct_example_fn_dir = os.path.join(correct_annotation_example_dir, 'stack_array_vein')
            correct_fn_list = os.listdir(correct_example_fn_dir)

    top_dict_rescaled_ct = data_top_dict + 'rescaled_ct-denoise/'
    fn_list = os.listdir(top_dict_rescaled_ct)[fold[0]::fold[1]]

    count = 0
    for fn in fn_list:

        print("processing:", fn, count, '/', len(fn_list))

        if os.path.exists(os.path.join(save_dict, fn)):
            print("processed")
            count += 1
            continue
        if correct_fn_list is not None:
            if fn not in correct_fn_list:
                print(fn, "not in correct file name list")
                count += 1
                continue
        blood_vessel_mask = np.load(vessel_mask_dict + fn)['array']

        raw_mask, model_output_gt, penalty_fn, _, penalty_fp = get_input_output_and_penalty_array(
            blood_vessel_mask, None)

        bounding_box = Functions.get_bounding_box(raw_mask, pad=0)
        print("bounding box:", bounding_box)

        x_min, x_max = bounding_box[0]
        y_min, y_max = bounding_box[1]
        z_min, z_max = bounding_box[2]
        print("original shape:", (x_max - x_min, y_max - y_min, z_max - z_min))

        if x_max - x_min < 256:
            x_min = max(0, int(x_min - (256 - (x_max - x_min)) / 2))
            x_max = x_min + 256
        if y_max - y_min < 256:
            y_min = max(0, int(y_min - (256 - (y_max - y_min)) / 2))
            y_max = y_min + 256
        if z_max - z_min < 256:
            z_min = max(0, int(z_min - (256 - (z_max - z_min)) / 2))
            z_max = z_min + 256

        print("padded shape:", (x_max - x_min, y_max - y_min, z_max - z_min), "new shape:", (256, 256, 256))

        raw_mask = rescale_to_new_shape(
            raw_mask[x_min: x_max, y_min: y_max, z_min: z_max], target_shape=(256, 256, 256))
        model_output_gt = rescale_to_new_shape(
            model_output_gt[x_min: x_max, y_min: y_max, z_min: z_max], target_shape=(256, 256, 256))
        penalty_fn = rescale_to_new_shape(
            penalty_fn[x_min: x_max, y_min: y_max, z_min: z_max], target_shape=(256, 256, 256))
        penalty_fp = rescale_to_new_shape(
            penalty_fp[x_min: x_max, y_min: y_max, z_min: z_max], target_shape=(256, 256, 256))

        stack_array = np.zeros([4, 256, 256, 256], 'float32')

        stack_array[0, :, :, :] = raw_mask
        stack_array[1, :, :, :] = model_output_gt
        stack_array[2, :, :, :] = penalty_fn
        stack_array[3, :, :, :] = penalty_fp

        Functions.save_np_array(save_dict, fn, stack_array, compress=True)
        count += 1


if __name__ == '__main__':
    get_stack_array_256(cta=False, fold=(0, 1), semantic='artery',
                        correct_annotation_example_dir='/data_disk/artery_vein_project/extract_blood_region/'
                                                       'training_data/sliced_sample/256_v0'
                        )
    get_stack_array_256(cta=False, fold=(0, 1), semantic='vein',
                        correct_annotation_example_dir='/data_disk/artery_vein_project/extract_blood_region/'
                                                       'training_data/sliced_sample/256_v0'
                        )
    exit()
