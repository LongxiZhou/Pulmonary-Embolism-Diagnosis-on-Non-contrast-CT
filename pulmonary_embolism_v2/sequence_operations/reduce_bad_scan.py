import os
from chest_ct_database.basic_functions import merge_dicts
import Tool_Functions.Functions as Functions


def remove_bad_sample_list_simple():
    fn_list_good = os.listdir('/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset/merged_v1/')
    fn_list = os.listdir('/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise/merged_v1/')

    for fn in fn_list:
        if fn not in fn_list_good:
            os.remove('/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise/merged_v1/' + fn)


def exclusion_based_on_report_dict(path_semantic_report, original_dataset_dict, save_dataset_dict, func_exclusion):
    """

    :param path_semantic_report: semantic report is a pickle object (dictionary) record information of relations of
    varies segmentation
    :param original_dataset_dict: pickle_object of sequences
    :param save_dataset_dict: if the sequence is good, save it to this dict
    :param func_exclusion: the function for check a sequence is good, in func_exclusion(semantic_report, sequence_name)
    :return:
    """

    semantic_report = Functions.pickle_load_object(path_semantic_report)

    sequence_name_list = os.listdir(original_dataset_dict)
    print("original there are", len(sequence_name_list), 'samples')

    saved_count = 0
    for sequence_name in sequence_name_list:
        if func_exclusion(semantic_report, sequence_name):
            continue
        Functions.copy_file(os.path.join(original_dataset_dict, sequence_name),
                            os.path.join(save_dataset_dict, sequence_name))
        saved_count += 1

    print("saved", saved_count, 'samples')

    return None


def func_exclusion_rad_dataset(semantic_report, sequence_name):
    """

    :param semantic_report: key is like 'trn000238'
    :param sequence_name: is like trn000238.pickle
    :return: False for Good Quality, True for bad quality
    """
    case_dict = semantic_report[sequence_name[:-7]]

    return func_exclusion_case_dict(case_dict, sequence_name)


def func_exclusion_case_dict(case_dict, sequence_name=None):
    """
    :return: False for Good Quality, True for bad quality
    """

    if case_dict['infection_to_lung_ratio'] > 0.075:
        if sequence_name is not None:
            print(sequence_name)
        print('bad sample: infection', case_dict['infection_to_lung_ratio'])
        return True
    if case_dict['blood_vessel_center_line_voxel'] < 6000:
        if sequence_name is not None:
            print(sequence_name)
        print('bad sample: blood center line', case_dict['blood_vessel_center_line_voxel'])
        return True
    key_list = list(case_dict.keys())

    blood_value_key_list = []
    for key in key_list:
        if 'average_blood HU' in key:
            blood_value_key_list.append(key)

    assert len(blood_value_key_list) > 0

    blood_value_list = []

    for blood_value_key in blood_value_key_list:
        blood_value_list.append(case_dict[blood_value_key])
    if max(blood_value_list) < 0 or min(blood_value_list) > 100:
        if sequence_name is not None:
            print(sequence_name)
        print('bad sample: blood average signal', blood_value_list)
        return True

    return False


def get_report_dict_merge():
    report_rad = Functions.pickle_load_object('/data_disk/RAD-ChestCT_dataset/report_dict.pickle')
    report_mudanjiang = Functions.pickle_load_object(
        '/data_disk/rescaled_ct_and_semantics/reports/COVID-19/mudanjiang/report_dict.pickle')
    report_yidayi = Functions.pickle_load_object(
        '/data_disk/rescaled_ct_and_semantics/reports/COVID-19/yidayi/report_dict.pickle')
    report_four_center = Functions.pickle_load_object(
        '/data_disk/rescaled_ct_and_semantics/reports/healthy_people/four_center_data/report_dict.pickle')
    report_xwzc = Functions.pickle_load_object(
        '/data_disk/rescaled_ct_and_semantics/reports/healthy_people/xwzc/report_dict.pickle')
    report_refine = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/report_dict.pickle')
    report_dict_merged = merge_dicts(
        [report_rad, report_mudanjiang, report_yidayi, report_four_center, report_xwzc, report_refine])
    return report_dict_merged


if __name__ == '__main__':
    report_merged = get_report_dict_merge()
    print(len(report_merged))

    sequence_dataset_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/temp_dataset_trim/'
    sequence_name_list = os.listdir(sequence_dataset_dict)

    print("initial_len:", len(sequence_name_list))
    num_bad = 0
    for name in sequence_name_list:

        if func_exclusion_rad_dataset(report_merged, name):
            num_bad += 1
            os.remove(os.path.join(sequence_dataset_dict, name))

    print(num_bad)
    print("number sequence good quality:", len(os.listdir(sequence_dataset_dict)))

    exit()
    exclusion_based_on_report_dict('/media/zhoul0a/New Volume/RAD-ChestCT_dataset/semantic_report_dict.pickle',
                                       '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise/RAD_3615/v1-3000/',
                                       '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise/merged_v1-3000/',
                                   func_exclusion_rad_dataset)

