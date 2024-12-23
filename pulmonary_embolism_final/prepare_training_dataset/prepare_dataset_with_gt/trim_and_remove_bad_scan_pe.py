"""
This dataset if for refining the model, and let the model know how blood clot looks like in non-contrast CT.

raw dataset is from non-contrast CT in CTA-non pairs.
CTA confirmed PE positive, and non-contrast is collected not so long before CTA
Thus, all these non-contrast CT should have very high probability of PE

remove the sample if:
it is not non-contrast (blood signal not in [0, 100])

remove some patch of the sample sequence to save GPU memory and control variance
remove patch if:
the branch level > 7
the length is > 1500 for low resolution and > 4000 for high resolution  (first remove patch with high branch level)
"""


import os
from chest_ct_database.basic_functions import merge_dicts
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import reconstruct_semantic_from_sample_sequence
import numpy as np


def get_report_dict_merge_pe_temp():
    import pe_dataset_management.basic_functions as basic_functions

    dataset_top_dict = '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast'

    report_dict_list = []

    for pe_dataset in basic_functions.get_dataset_relative_path('PE'):
        path_report = os.path.join(dataset_top_dict, pe_dataset, 'reports', 'report_dict.pickle')
        report_dict_list.append(Functions.pickle_load_object(path_report))
    for pe_dataset in basic_functions.get_dataset_relative_path('Temp'):
        path_report = os.path.join(dataset_top_dict, pe_dataset, 'reports', 'report_dict.pickle')
        if os.path.exists(path_report):
            report_dict_list.append(Functions.pickle_load_object(path_report))

    report_dict_merged = merge_dicts(report_dict_list, check_overlap_keys=False)
    return report_dict_merged


def reduce_sequence_length(sample_sequence, func_exclusion=None, target_length=4000, max_branch=9):
    """

    :param max_branch: remove a cube if it exceed max_branch
    :param sample_sequence: a list of samples
    :param func_exclusion: input a sample, output a float, higher means first to exclude
    :param target_length:
    :return: sample_sequence_copy of target_length
    """

    if func_exclusion is None:
        func_exclusion = exclusion_large_branch

    original_len = len(sample_sequence)

    if len(sample_sequence) < target_length:
        new_sample_sequence = []
        for sample in sample_sequence:
            if sample['branch_level'] >= max_branch:
                continue
            else:
                new_sample_sequence.append(sample)

        print("original length:", original_len, "length after trim:", len(new_sample_sequence))
        return new_sample_sequence

    remove_number = len(sample_sequence) - target_length

    index_exclusion_list = []

    for index in range(len(sample_sequence)):
        sample = sample_sequence[index]
        index_exclusion_list.append((index, func_exclusion(sample)))

    def func_compare(item_1, item_2):
        if item_1[1] > item_2[1]:
            return 1
        return -1

    index_exclusion_list = Functions.customized_sort(index_exclusion_list, func_compare, True)

    list_remove_index = []
    for i in range(remove_number):
        list_remove_index.append(index_exclusion_list[i][0])

    new_sample_sequence = []
    for index in range(original_len):
        if index in list_remove_index:
            continue
        if sample_sequence[index]['branch_level'] >= max_branch:
            continue
        new_sample_sequence.append(sample_sequence[index])

    print("original length:", original_len, "length after trim:", len(new_sample_sequence))

    return new_sample_sequence


def exclusion_large_branch(sample):
    return sample['branch_level']


def get_blood_signal_hu(case_dict):
    key_list = list(case_dict.keys())

    blood_value_key_list = []
    for key in key_list:
        if 'average_blood HU' in key:
            blood_value_key_list.append(key)

    assert len(blood_value_key_list) > 0

    blood_value_list = []

    for blood_value_key in blood_value_key_list:
        blood_value_list.append(case_dict[blood_value_key])
    return blood_value_list


def func_exclusion_case_dict(case_dict, sequence_name=None, cta=False):
    """
    case_dict = report_dict_merged[sequence_name[:-7]]
    :return: False for Good Quality, True for bad quality
    """

    blood_value_list = get_blood_signal_hu(case_dict)

    if not cta:
        if max(blood_value_list) < 0 or min(blood_value_list) > 100:
            if sequence_name is not None:
                print(sequence_name)
            print('bad sample for non-contrast: blood average signal', blood_value_list)
            return True
    else:
        if min(blood_value_list) <= 100:
            if sequence_name is not None:
                print(sequence_name)
            print('bad sample for CTA: blood average signal', blood_value_list)
            return True

    return False


def trim_and_reduce_for_dataset(dict_original_dataset, dict_save_new_dataset,
                                high_resolution=True, fold=(0, 1), reprocess=True, target_length=4000, max_branch=9,
                                cta=False, report_dict_merged=None):
    """
    :param cta: whether the dataset is CTA dataset
    :param max_branch: the max branch of cube
    :param target_length: the max length of sequence
    :param reprocess: if True, overwrite existing files
    :param fold:
    :param high_resolution:
    :param dict_original_dataset:
    :param dict_save_new_dataset:
    :param report_dict_merged
    :return: None
    """
    if not os.path.exists(dict_original_dataset):
        print("original dataset:", dict_original_dataset, "not exist")
        return None
    sample_name_list = os.listdir(dict_original_dataset)
    sample_name_list = Functions.split_list_by_ord_sum(sample_name_list, fold=fold)
    print("there are", len(sample_name_list), "samples in the dataset")

    processed_count = 0
    if report_dict_merged is None:
        report_dict_merged = get_report_dict_merge_pe_temp()

    for sample_name in sample_name_list:
        print("processing:", sample_name, processed_count, '/', len(sample_name_list))
        save_path = os.path.join(dict_save_new_dataset, sample_name)
        if os.path.exists(save_path):
            if reprocess:
                print("overwrite path:", save_path)
            else:
                print("processed")
                processed_count += 1
                continue

        if func_exclusion_case_dict(report_dict_merged[sample_name[:-7]], sample_name, cta=cta):
            # this means this sample is not good
            processed_count += 1
            continue

        sample_object = Functions.pickle_load_object(os.path.join(dict_original_dataset, sample_name))

        sample_object_trimmed = trim_sample(
            sample_object, high_resolution=high_resolution, target_length=target_length, max_branch=max_branch)

        Functions.pickle_save_object(save_path, sample_object_trimmed)
        processed_count += 1


def trim_sample(sample_object, high_resolution=True, target_length=None, max_branch=9):
    """

    :return: trim_sample_object
    """
    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
        if target_length is None:
            target_length = 1500
    else:
        absolute_cube_length = (4, 4, 5)
        if target_length is None:
            target_length = 4000

    original_sequence = sample_object["sample_sequence"]
    original_blood_center_line = np.zeros([512, 512, 512], 'float32')
    original_blood_center_line[sample_object["center_line_loc_array"]] = 1
    trimmed_sequence = reduce_sequence_length(original_sequence, target_length=target_length, max_branch=max_branch)
    blood_region_in_trimmed_sequence = reconstruct_semantic_from_sample_sequence(
        trimmed_sequence, absolute_cube_length, key="depth_cube")
    blood_region_in_trimmed_sequence = np.array(blood_region_in_trimmed_sequence > 0.5, 'float16')
    trimmed_center_line = original_blood_center_line * blood_region_in_trimmed_sequence

    sample_object["center_line_loc_array"] = np.where(trimmed_center_line > 0.5)
    sample_object["sample_sequence"] = trimmed_sequence

    return sample_object


if __name__ == '__main__':
    exit()
