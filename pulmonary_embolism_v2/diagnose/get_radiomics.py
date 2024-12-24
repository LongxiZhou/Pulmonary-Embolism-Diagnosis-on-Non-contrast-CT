
import numpy as np
import Tool_Functions.Functions as Functions
import pulmonary_embolism_v2.sequence_rescaled_ct_converter as converter
import math
import os
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as clot_prediction


def artery_vein_clot_radiomics(rescaled_ct, sample_sequence_clot_predicted, artery_mask, vein_mask,
                               depth_array, vessel_center_line, lung_mask, branch_array, min_depth=2):
    """

    :param branch_array:
    :param lung_mask:
    :param rescaled_ct
    :param sample_sequence_clot_predicted:
    :param artery_mask:
    :param vein_mask:
    :param depth_array:
    :param vessel_center_line
    :param min_depth:
    :return: {radiomic-name: (artery_vein_difference_normalized_by_vein_std, artery_vein_difference)}

    """

    radiomic_dict = {}

    predicted_clot_certainty = \
        converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence_clot_predicted, (4, 4, 5),
                                                               key='clot_certainty_mask')
    # probability for clot is: exp(clot_certainty) / (1 + exp(clot_certainty))

    predicted_clot_probability = \
        converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence_clot_predicted, (4, 4, 5),
                                                               key='clot_prob_mask')

    valid_region = np.array(rescaled_ct > 3 / 8, 'float32') * np.array(depth_array >= min_depth, 'float32')
    # HU value > 0 and depth >= 2

    artery_valid_mask = valid_region * artery_mask
    vein_valid_mask = valid_region * vein_mask

    artery_center_line_mask = vessel_center_line * artery_valid_mask
    vein_center_line_mask = vessel_center_line * vein_valid_mask

    mass_center_y = get_mass_center(sample_sequence_clot_predicted)[1]

    inside_lung_ct = rescaled_ct * lung_mask
    ct_value_av_difference_left_lung, ct_value_av_difference_right_lung = get_pe_statistics(
        inside_lung_ct * artery_valid_mask, inside_lung_ct * vein_valid_mask, sparse=True, mass_center_y=mass_center_y)

    radiomic_dict["a-v_ct_value_difference-left_inside-lung"] = ct_value_av_difference_left_lung
    radiomic_dict["a-v_ct_value_difference-right_inside-lung"] = ct_value_av_difference_right_lung

    half_strip_mask = reduce_by_half_depth(branch_array, depth_array, min_depth)
    half_strip_ct = rescaled_ct * half_strip_mask
    ct_value_av_difference_left, ct_value_av_difference_right = get_pe_statistics(
        half_strip_ct * artery_valid_mask, half_strip_ct * vein_valid_mask, sparse=True, mass_center_y=mass_center_y)

    radiomic_dict["a-v_ct_value_difference-left_half-strip"] = ct_value_av_difference_left
    radiomic_dict["a-v_ct_value_difference-right_half-strip"] = ct_value_av_difference_right

    certainty_difference_overall = get_pe_statistics(
        predicted_clot_certainty * artery_valid_mask, predicted_clot_certainty * vein_valid_mask, sparse=True)
    radiomic_dict["certainty_strip-one-layer"] = certainty_difference_overall

    half_strip_certainty = predicted_clot_certainty * half_strip_mask
    certainty_difference_half_strip = get_pe_statistics(
        half_strip_certainty * artery_valid_mask, half_strip_certainty * vein_valid_mask, sparse=True)
    radiomic_dict["certainty_half-strip"] = certainty_difference_half_strip

    center_line_certainty_artery = artery_center_line_mask * predicted_clot_certainty
    center_line_certainty_vein = vein_center_line_mask * predicted_clot_certainty
    certainty_difference_center_line = get_pe_statistics(center_line_certainty_artery, center_line_certainty_vein,
                                                         sparse=True)
    radiomic_dict["certainty_center-line"] = certainty_difference_center_line

    probability_difference_overall = get_pe_statistics(
        predicted_clot_probability * artery_valid_mask, predicted_clot_probability * vein_valid_mask, sparse=True)
    radiomic_dict["probability_strip-one-layer"] = probability_difference_overall

    half_strip_probability = predicted_clot_probability * half_strip_mask
    probability_difference_half_strip = get_pe_statistics(
        half_strip_probability * artery_valid_mask, half_strip_probability * vein_valid_mask, sparse=True)
    radiomic_dict["probability_half-strip"] = probability_difference_half_strip

    center_line_prob_artery = artery_center_line_mask * predicted_clot_probability
    center_line_prob_vein = vein_center_line_mask * predicted_clot_probability
    probability_difference_center_line = get_pe_statistics(center_line_prob_artery, center_line_prob_vein,
                                                           sparse=True)
    radiomic_dict["probability_center-line"] = probability_difference_center_line

    return radiomic_dict


def pipeline_process(top_dict_rescaled_ct, top_dict_sample_sequence, top_dict_semantics, top_dict_center_line_and_depth,
                     report_save_path, basic_tissue_report_dict_path, model_path=None, min_depth=2,
                     dataset_class='non_clot', list_bad_scan_name=None):

    # report_dict with structure:
    # {class: {patient-id:
    # {clot_radiomics: {radiomic-name: (normalized, not_normalized)}, basic_tissue_radiomics: {radiomic-name: value}}}}

    assert dataset_class in ['non_clot', 'with_clot', 'blind_dataset']
    if os.path.exists(report_save_path):
        report_dict = Functions.pickle_load_object(report_save_path)
    else:
        report_dict = {'non_clot': {}, 'with_clot': {}, 'blind_dataset': {}}

    processed_file_name_list = list(report_dict[dataset_class].keys())
    file_name_list = os.listdir(top_dict_rescaled_ct)
    transformer_model = clot_prediction.load_saved_model_guided(model_path)
    basic_tissue_report_dict = Functions.pickle_load_object(basic_tissue_report_dict_path)

    def process_one(file_name):  # here file_name should be like  .npy or .npz
        if file_name[-1] == 'z':
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, file_name))['array']
        else:
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, file_name))
        sample_sequence = Functions.pickle_load_object(
            os.path.join(top_dict_sample_sequence, file_name[:-4] + '.pickle'))
        sample_sequence_clot_predicted = \
            clot_prediction.predict_clot_for_sample_sequence(
                sample_sequence, model=transformer_model, min_depth=min_depth)

        artery_mask = np.load(os.path.join(top_dict_semantics, 'artery_mask', file_name))
        artery_mask = artery_mask[list(artery_mask.keys())[0]]
        vein_mask = np.load(os.path.join(top_dict_semantics, 'vein_mask', file_name))
        vein_mask = vein_mask[list(vein_mask.keys())[0]]

        lung_mask = np.load(os.path.join(top_dict_semantics, 'lung_mask', file_name))['array']
        depth_array = np.load(os.path.join(top_dict_center_line_and_depth, 'depth_array', file_name))['array']

        def get_blood_average_value():
            sample_loc_list = Functions.get_location_list(np.where(depth_array > 0.5 * np.max(depth_array)))
            sample_value_list = []
            for loc in sample_loc_list:
                sample_value_list.append(rescaled_ct[loc])
            return np.average(sample_value_list) * 1600 - 600

        blood_value_average = get_blood_average_value()
        if not 0 < blood_value_average < 100:
            print("the blood value for file name is strange:", blood_value_average)
            raise ValueError

        vessel_center_line = \
            np.load(os.path.join(top_dict_center_line_and_depth, 'blood_center_line', file_name))['array']
        branch_array = np.load(os.path.join(top_dict_center_line_and_depth, 'blood_branch_map', file_name))['array']

        sample_radiomic_dict = artery_vein_clot_radiomics(rescaled_ct, sample_sequence_clot_predicted, artery_mask,
                                                          vein_mask, depth_array, vessel_center_line, lung_mask,
                                                          branch_array, min_depth)
        return sample_radiomic_dict

    total_sample = len(file_name_list)
    processed_count = 0

    if list_bad_scan_name is None:
        list_bad_scan_name = []

    for name in file_name_list:
        print("processing:", name[:-4], processed_count, '/', total_sample)
        if name[:-4] in processed_file_name_list:
            print("processed.")
            processed_count += 1
            continue

        if name[:-4] in list_bad_scan_name:
            print("bad scan")
            processed_count += 1
            continue

        dict_clot_radiomics = process_one(name)
        dict_basic_tissue_radiomics = basic_tissue_report_dict[name[:-4]]

        report_dict[dataset_class][name[:-4]] = \
            {'clot_radiomics': dict_clot_radiomics, 'basic_tissue_radiomics': dict_basic_tissue_radiomics}

        print("clot radiomics:")
        print(dict_clot_radiomics)
        print("basic_tissue_radiomics:")
        print(dict_basic_tissue_radiomics)

        Functions.pickle_save_object(report_save_path, report_dict)
        processed_count += 1


def get_pe_statistics(artery_info_array, vein_info_array, sparse=True, mass_center_y=None, normalize='both'):
    """

    :param artery_info_array:
    :param vein_info_array:
    :param sparse: whether the non zero is sparse
    :param mass_center_y: is not None, split it into left and right lung
    :param normalize: whether normalize by the std of the vein
    :return: difference between artery and vein
    """

    if mass_center_y is None:
        sorted_array_artery, artery_non_zeros = \
            Functions.sort_non_zero_voxels(artery_info_array, sparse=sparse, reverse=False)
        sorted_array_vein, vein_non_zeros = \
            Functions.sort_non_zero_voxels(vein_info_array, sparse=sparse, reverse=False)

        assert artery_non_zeros * vein_non_zeros > 0
        print("artery_non_zeros:", artery_non_zeros, "vein_non_zeros:", vein_non_zeros)

        median_artery = sorted_array_artery[int(artery_non_zeros / 2)]  # the normal value for artery
        median_vein = sorted_array_vein[int(vein_non_zeros / 2)]  # the normal value for vein

        average_upper_half_artery = np.average(sorted_array_artery[int(artery_non_zeros / 2): artery_non_zeros])
        average_upper_half_vein = np.average(sorted_array_vein[int(vein_non_zeros / 2): vein_non_zeros])

        a_v_difference = (average_upper_half_artery - median_artery) - (average_upper_half_vein - median_vein)

        if normalize == 'both':
            std_vein = np.std(sorted_array_vein[0: vein_non_zeros])  # normalize by the data quality
            return a_v_difference / std_vein, a_v_difference

        if normalize:
            std_vein = np.std(sorted_array_vein[0: vein_non_zeros])  # normalize by the data quality
            return a_v_difference / std_vein
        return a_v_difference
    else:
        left_mask = np.ones(np.shape(artery_info_array), 'float32')
        left_mask[:, 0: mass_center_y, :] = 0
        right_mask = 1 - left_mask

        artery_info_array_left = artery_info_array * left_mask
        vein_info_array_left = vein_info_array * left_mask

        artery_info_array_right = artery_info_array * right_mask
        vein_info_array_right = vein_info_array * right_mask

        return get_pe_statistics(
            artery_info_array_left, vein_info_array_left, sparse, None), get_pe_statistics(
            artery_info_array_right, vein_info_array_right, sparse, None)


def get_mass_center(sample_sequence):
    center_location = sample_sequence[0]['center_location']
    location_offset = sample_sequence[0]['location_offset']

    mass_center = (center_location[0] - location_offset[0], center_location[1] - location_offset[1],
                   center_location[2] - location_offset[2])

    return mass_center


def reduce_by_half_depth(branch_array, depth_array, min_depth=2):
    depth_nearest_center_line = np.exp(branch_array * math.log(0.7)) * 30 * np.array(depth_array > 0.5, 'float32')

    half_depth_mask = np.array(depth_array * 2 > depth_nearest_center_line, 'float32')

    half_depth_mask = half_depth_mask * np.array(depth_array >= min_depth, 'float32')

    return half_depth_mask


if __name__ == '__main__':
    pipeline_process('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct_denoise/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/'
                     'scan_without_clot/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/basic_semantics/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/depth_and_center-line/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/diagnose/artery_vein_clot_radiomics.pickle',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/report_dict.pickle',
                     dataset_class='non_clot',
                     list_bad_scan_name=['patient-id-S81190', 'patient-id-S45340',
                                         'patient-id-447', 'patient-id-22239200'])

    exit()
