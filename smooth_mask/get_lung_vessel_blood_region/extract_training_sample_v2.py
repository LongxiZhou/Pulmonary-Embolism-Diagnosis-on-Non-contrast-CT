"""
Vessels in CTA has high average HU value, usually higher than 200, which is significantly higher than adjacent tissue.

version 2 only for CTA with high blood vessel CT signals (> 200)

In this version, we first cast CTA to non-contrast, then get semantic analysis, which will be more precise

source file from:
/data_disk/CTA-CT_paired-dataset/dataset_CTA

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
import os
from format_convert.spatial_normalize import rescale_to_new_shape
from smooth_mask.get_lung_vessel_blood_region.extract_blood_region_and_training_sample import \
    get_input_output_and_penalty_array
from pe_dataset_management.basic_functions import get_dataset_relative_path


def process_sub_dataset_v2(rescaled_ct_dict, mask_dict, mask_depth_dict, branch_dict, save_dict,
                           path_wrong_file_name_list, fold=(0, 1)):
    if not os.path.exists(rescaled_ct_dict):
        return None
    fn_list = os.listdir(rescaled_ct_dict)[fold[0]::fold[1]]
    count = 0

    if os.path.exists(path_wrong_file_name_list):
        wrong_file_name_list = Functions.pickle_load_object(path_wrong_file_name_list)
    else:
        wrong_file_name_list = []

    for fn in fn_list:

        print("processing:", fn, count, '/', len(fn_list))
        if os.path.exists(os.path.join(save_dict, fn)):
            print("processed")
            count += 1
            continue

        if fn in wrong_file_name_list:
            print("not qualified file name for", fn)
            count += 1
            continue

        rescaled_ct_denoise = np.load(os.path.join(rescaled_ct_dict, fn))['array']
        depth_array = np.load(os.path.join(mask_depth_dict, fn))['array']
        ave_ct = Functions.change_to_HU(
            np.average(rescaled_ct_denoise[np.where(depth_array > 0.5 * np.max(depth_array))]))
        print("average blood HU:", ave_ct)
        if not ave_ct >= 200:
            print("this file does not have very high blood signal, pass it")
            wrong_file_name_list.append(fn)
            Functions.pickle_save_object(path_wrong_file_name_list, wrong_file_name_list)
            count += 1
            continue

        blood_branch_map = np.load(os.path.join(branch_dict, fn))['array']
        blood_vessel_mask = np.load(os.path.join(mask_dict, fn))['array']

        raw_mask, model_output_gt, penalty_fn, _, penalty_fp = get_input_output_and_penalty_array(
            rescaled_ct_denoise, blood_branch_map, None, blood_vessel_mask, None, version='2', apply_cta_threshold=True)

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


def get_relevant_directories(database_cta='/data_disk/CTA-CT_paired-dataset/dataset_CTA',
                             sub_dataset_dir='PE_High_Quality', semantic='artery',
                             save_top_dict='/data_disk/artery_vein_project/extract_blood_region/'
                                           'training_data/sliced_sample/256_v2/CTA/'):

    # CT use CTA
    rescaled_ct_dict = os.path.join(database_cta, sub_dataset_dir, 'rescaled_ct-denoise')
    path_wrong_file_name_list = os.path.join(
        '/data_disk/artery_vein_project/extract_blood_region/training_data/reports', 'not_qualified_fn_list',
        sub_dataset_dir, semantic + '.pickle')

    # semantics use simulated non-contrast from CTA
    assert semantic in ['artery', 'vein', 'blood', 'blood_high_recall']
    if semantic == 'artery':
        save_dict = os.path.join(save_top_dict, 'stack_array_artery')
        branch_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                   'depth_and_center-line/artery_branch_map')
        mask_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast', 'semantics/artery_mask')
        mask_depth_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                       'depth_and_center-line/depth_array_artery')
    elif semantic == 'vein':
        save_dict = os.path.join(save_top_dict, 'stack_array_vein')
        branch_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                   'depth_and_center-line/vein_branch_map')
        mask_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast', 'semantics/vein_mask')
        mask_depth_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                       'depth_and_center-line/depth_array_vein')
    elif semantic == 'blood':
        save_dict = os.path.join(save_top_dict, 'stack_array_blood')
        branch_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                   'depth_and_center-line/blood_branch_map')
        mask_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast', 'semantics/blood_mask')
        mask_depth_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                       'depth_and_center-line/depth_array')
    else:
        assert semantic == 'blood_high_recall'
        save_dict = os.path.join(save_top_dict, 'stack_array_blood_high_recall')
        branch_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                   'depth_and_center-line/high_recall_blood_branch_map')
        mask_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                 'semantics/blood_mask_high_recall')
        mask_depth_dict = os.path.join(database_cta, sub_dataset_dir, 'simulated_non_contrast',
                                       'depth_and_center-line/high_recall_depth_array')

    return rescaled_ct_dict, mask_dict, mask_depth_dict, branch_dict, save_dict, path_wrong_file_name_list


def get_stack_array_256_version2(top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA',
                                 save_top_dict='/data_disk/artery_vein_project/extract_blood_region/'
                                               'training_data/sliced_sample/256_v2/CTA/', fold=(0, 1)):

    # only for CTA with blood vessel signal > 200 HU
    # blood region is defined as blood mask with > 100 HU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0] % 2)

    list_sub_dataset = get_dataset_relative_path()
    semantic_list = ['blood_high_recall', 'blood', 'artery', 'vein']

    for semantic in semantic_list:
        for sub_dataset in list_sub_dataset:
            rescaled_ct_dict, mask_dict, mask_depth_dict, branch_dict, save_dict, path_wrong_file_name_list = \
                get_relevant_directories(top_dict_database, sub_dataset, semantic, save_top_dict)
            process_sub_dataset_v2(rescaled_ct_dict, mask_dict, mask_depth_dict, branch_dict, save_dict,
                                   path_wrong_file_name_list, fold=fold)


if __name__ == '__main__':
    get_stack_array_256_version2(fold=(0, 1))

    exit()
