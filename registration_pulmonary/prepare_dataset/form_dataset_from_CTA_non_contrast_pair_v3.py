"""
top_directory: /data_disk/CTA-CT_paired-dataset

sample is in .npy float16 format,
in shape [7, 256, 256, 256]

channel 0 is the normalized ct fix (non-contrast), in numpy float16 shaped [256, 256, 256] ,
mass center of blood vessel set to (128, 128, 128)

channel 1 is the vessel depth array (max_depth_normalized to 1) for fix (non-contrast CT)

channel 2 is the vessel branch array for fix (non-contrast CT)

channel 3 is the normalized ct moving (CTA in simulated non-contrast), numpy float16 shaped [256, 256, 256] ,
mass center of blood vessel set to (128, 128, 128)

channel 4 is the vessel depth array (max_depth_normalized to 1) for moving (CTA in simulated non-contrast)

channel 5 is the vessel branch array for moving (CTA in simulated non-contrast)

channel 6 is the penalty weights for ncc loss based on non-contrast, numpy float16 shaped [256, 256, 256]

The task is to register CTA (simulated non-contrast) to non-contrast
"""

import os
import numpy as np
import format_convert.spatial_normalize as spatial_normalize
import format_convert.basic_transformations as basic_transformations
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import registration_pulmonary.prepare_dataset.form_ncc_penalty_weight as get_ncc_penalty
import pe_dataset_management.basic_functions as pe_dataset_funcs


def add_landmark_for_blood_vessel(rescaled_ct_denoise, depth_array_blood_vessel_high_recall, max_depth_normalize=1):
    """
    direct use the CT image for registration is not a good idea.
    here we add extra landmarks to help registration

    :param rescaled_ct_denoise:
    :param depth_array_blood_vessel_high_recall:
    :param max_depth_normalize:
    :return:
    """
    normalize_divisor = np.max(depth_array_blood_vessel_high_recall) / max_depth_normalize
    normalized_depth = depth_array_blood_vessel_high_recall / normalize_divisor
    return rescaled_ct_denoise + normalized_depth


def process_one_scan_name(scan_name, top_dict_pe_dataset='/data_disk/CTA-CT_paired-dataset',
                          top_dict_save='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_256',
                          show=False):
    if len(scan_name) < 4:
        scan_name = scan_name + '.npz'
    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    top_dict_cta, top_dict_non_contrast = pe_dataset_funcs.find_patient_id_dataset_correspondence(
        scan_name, strip=True, top_dict=top_dict_pe_dataset)

    vessel_mask_path_cta = os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'semantics', 'blood_mask_high_recall', scan_name)
    vessel_mask_cta = np.load(vessel_mask_path_cta)['array']
    vessel_normalized_cta, flow_cta = \
        basic_transformations.down_sample_central_mass_center_and_crop_size(vessel_mask_cta, crop=False)

    rescaled_cta = np.load(os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'rescaled_ct-denoise', scan_name))['array']

    landmark_path_cta_1 = os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'depth_and_center-line', 'high_recall_depth_array', scan_name)
    landmark_cta_1 = np.load(landmark_path_cta_1)['array']
    landmark_cta_1 = landmark_cta_1 / np.max(landmark_cta_1)

    landmark_path_cta_2 = os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'depth_and_center-line', 'high_recall_blood_branch_map', scan_name)
    landmark_cta_2 = np.load(landmark_path_cta_2)['array']

    rescaled_cta = add_landmark_for_blood_vessel(rescaled_cta, landmark_cta_1, max_depth_normalize=2)

    normalized_cta = basic_transformations.transformation_on_array(rescaled_cta, flow_cta)
    normalized_landmark_cta_1 = basic_transformations.transformation_on_array(landmark_cta_1, flow_cta)
    normalized_landmark_cta_2 = basic_transformations.transformation_on_array(landmark_cta_2, flow_cta)

    vessel_mask_path_non = os.path.join(
        top_dict_non_contrast, 'semantics', 'blood_mask_high_recall', scan_name)
    vessel_mask_non = np.load(vessel_mask_path_non)['array']
    vessel_normalized_non, flow_non = \
        basic_transformations.down_sample_central_mass_center_and_crop_size(vessel_mask_non, crop=False)

    rescaled_non = np.load(os.path.join(top_dict_non_contrast, 'rescaled_ct-denoise', scan_name))['array']

    landmark_path_non_1 = os.path.join(
        top_dict_non_contrast, 'depth_and_center-line', 'high_recall_depth_array', scan_name)
    landmark_non_1 = np.load(landmark_path_non_1)['array']
    landmark_non_1 = landmark_non_1 / np.max(landmark_non_1)

    landmark_path_non_2 = os.path.join(
        top_dict_non_contrast, 'depth_and_center-line', 'high_recall_blood_branch_map', scan_name)
    landmark_non_2 = np.load(landmark_path_non_2)['array']

    rescaled_non = add_landmark_for_blood_vessel(rescaled_non, landmark_non_1, max_depth_normalize=2)

    normalized_non = basic_transformations.transformation_on_array(rescaled_non, flow_non)
    normalized_landmark_non_1 = basic_transformations.transformation_on_array(landmark_non_1, flow_non)
    normalized_landmark_non_2 = basic_transformations.transformation_on_array(landmark_non_2, flow_non)

    # use non-contrast to form penalty.
    ncc_penalty = get_ncc_penalty.calculate_penalty_weight_pe_paired_dataset(scan_name, top_dict_pe_dataset, show=False)

    normalized_ncc_penalty = basic_transformations.transformation_on_array(ncc_penalty, flow_non)

    final_array = np.zeros([7, 256, 256, 256], 'float16')
    final_array[0, :, :, :] = normalized_non
    final_array[1, :, :, :] = normalized_landmark_non_1
    final_array[2, :, :, :] = normalized_landmark_non_2
    final_array[3, :, :, :] = normalized_cta
    final_array[4, :, :, :] = normalized_landmark_cta_1
    final_array[5, :, :, :] = normalized_landmark_cta_2
    final_array[6, :, :, :] = normalized_ncc_penalty

    if show:
        print('original vessel dice:', metrics.dice_score_two_class(vessel_mask_cta, vessel_mask_non))
        show_sample(final_array, 128, gray=False)
        print('normalized vessel dice:', metrics.dice_score_two_class(vessel_normalized_cta, vessel_normalized_non))

    Functions.save_np_array(top_dict_save, scan_name[:-4] + '.npy', final_array, compress=False)

    return final_array


def process_all_scan_all_scale(top_dict_pe_dataset='/data_disk/CTA-CT_paired-dataset',
                               top_dict_256='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_256',
                               top_dict_128='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_128',
                               top_dict_64='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_64',
                               top_dict_32='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_32',
                               fold=(0, 1)):
    fn_list = pe_dataset_funcs.get_all_scan_name(top_dict_pe_dataset)[fold[0]:: fold[1]]

    if not os.path.exists(top_dict_256):
        os.makedirs(top_dict_256)
    processed_name = os.listdir(top_dict_256)

    processed_count = 0

    wrong_file_name_list = ['Z108']

    for fn in fn_list:
        print("processing:", fn, processed_count, '/', len(fn_list))
        if fn in wrong_file_name_list:
            print("wrong scan")
            processed_count += 1
            continue
        if fn + '.npy' in processed_name:
            print('processed')
            processed_count += 1
            continue
        process_one_scan_name(fn, top_dict_pe_dataset, top_dict_256, show=False)
        processed_count += 1

    form_data_down_sampled(top_dict_256, top_dict_128, top_dict_64, top_dict_32, fold)


def form_data_down_sampled(top_dict_256='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_256',
                           top_dict_128='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_128',
                           top_dict_64='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_64',
                           top_dict_32='/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_32',
                           fold=(0, 1)):
    scan_name_list = os.listdir(top_dict_256)[fold[0]:: fold[1]]

    def process_one_down_sample(target_dict, final_cube_length):
        processed_count = 0
        if os.path.exists(target_dict):
            processed_name_list = os.listdir(target_dict)
        else:
            processed_name_list = []
        for scan_name in scan_name_list:
            print("processing", scan_name, processed_count, '/', len(scan_name_list))
            if scan_name in processed_name_list:
                print("processed")
                processed_count += 1
                continue

            original_sample = np.load(os.path.join(top_dict_256, scan_name))
            new_sample = np.zeros([7, final_cube_length, final_cube_length, final_cube_length], 'float16')

            new_shape = (final_cube_length, final_cube_length, final_cube_length)

            for i in range(7):
                new_sample[i, :, :, :] = spatial_normalize.rescale_to_new_shape(
                    original_sample[i, :, :, :], new_shape, change_format=True)

            Functions.save_np_array(target_dict, scan_name, new_sample, compress=False)
            processed_count += 1

    process_one_down_sample(top_dict_128, 128)
    process_one_down_sample(top_dict_64, 64)
    process_one_down_sample(top_dict_32, 32)


def show_sample(sample_array, z_loc=128, gray=False):
    img_1 = sample_array[0, :, :, z_loc]
    img_2 = sample_array[1, :, :, z_loc]
    img_3 = sample_array[2, :, :, z_loc]
    img_4 = sample_array[3, :, :, z_loc]
    img_5 = sample_array[4, :, :, z_loc]
    img_6 = sample_array[5, :, :, z_loc]
    img_7 = sample_array[6, :, :, z_loc]

    Functions.show_multiple_images(img1=img_1, img2=img_2, img3=img_3,
                                   img4=img_4, img5=img_5, img6=img_6,
                                   img7=img_7, gray=gray)


if __name__ == '__main__':

    process_all_scan_all_scale(fold=(0, 1))
    exit()
    process_one_scan_name('Z111.npz', show=True)
    exit()
    sample = np.load('/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_256/11.17p05.npy')

    show_sample(sample, 128)

    exit()

    form_data_down_sampled(fold=(0, 1))
    exit()
