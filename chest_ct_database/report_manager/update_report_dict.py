import os
import warnings
import numpy as np
import Tool_Functions.Functions as Functions
from chest_ct_database.basic_functions import merge_dicts, extract_relative_dirs_sub_dataset


"""
func_update_report(key_name, list_reference_dict, report_dict=None)
                here the function use data in reference dict to update the report_dict[key_name],
                return a case dict for the key_name
func_check_processed(report_dict, key_name), return True if processed, False for not
                processed
"""


def update_report_dict_one_dataset(top_dict_for_file_names, report_save_path, func_update_report, func_check_processed,
                                   list_reference_dict=None):
    """

    :param top_dict_for_file_names: this dict only contains file name
    :param report_save_path:
    :param func_update_report: func_update_report(key_name, list_reference_dict, report_dict=None)
                here the function use data in reference dict to update the report_dict[key_name],
                return a case dict for the key_name
    :param func_check_processed: func_check_processed(report_dict, key_name), return True if processed, False for not
                processed
    :param list_reference_dict: pass to func_update_report
    :return:
    """

    assert os.path.exists(report_save_path)
    report_dict = Functions.pickle_load_object(report_save_path)

    file_name_list = os.listdir(top_dict_for_file_names)

    key_list = list(report_dict.keys())

    processed_count = 0

    for file_name in file_name_list:

        print("processing:", file_name[:-4], processed_count, '/', len(file_name_list))

        if file_name[:-4] not in key_list:
            warnings.warn("file name not as the key in report dict")
            processed_count += 1
            continue

        if func_check_processed(report_dict, file_name[:-4]):
            print("processed")
            processed_count += 1
            continue

        case_dict = report_dict[file_name[:-4]]

        updated_dict = func_update_report(file_name[:-4], list_reference_dict, report_dict)

        print("update_dict:")
        print(updated_dict)

        updated_dict = merge_dicts([case_dict, updated_dict])

        report_dict[file_name[:-4]] = updated_dict

        Functions.pickle_save_object(report_save_path, report_dict)

        processed_count += 1


def update_report_dict_database(top_dict_source, top_dict_report, func_update_report, func_check_processed,
                                list_reference_feature_top_dict=None):
    """

    :param top_dict_source:
    :param top_dict_report:
    :param func_update_report:
    :param func_check_processed:
    :param list_reference_feature_top_dict:
    :return:
    """
    list_dataset_dirs = extract_relative_dirs_sub_dataset(top_dict_source)

    processed_count = 0

    for sub_dataset in list_dataset_dirs:
        print("\n#########################################")
        print("updating report for", sub_dataset, processed_count, '/', len(list_dataset_dirs))
        print("#########################################")

        top_dict_for_file_names = os.path.join(top_dict_source, sub_dataset)

        report_save_path = os.path.join(top_dict_report, sub_dataset, 'report_dict.pickle')

        if list_reference_feature_top_dict is not None:
            list_reference_dict = []
            for reference_feature_top_dict in list_reference_feature_top_dict:
                list_reference_dict.append(os.path.join(reference_feature_top_dict, sub_dataset))
        else:
            list_reference_dict = None

        update_report_dict_one_dataset(top_dict_for_file_names, report_save_path, func_update_report,
                                       func_check_processed, list_reference_dict)

        processed_count += 1


def update_inherent_noise_one_dataset(top_dict_rescaled_ct, top_dict_semantics, report_save_path, denoise=True,
                                      version=None):
    """

    :param version: version for denoise model
    :param top_dict_rescaled_ct:
    :param top_dict_semantics:
    :param report_save_path:
    :param denoise:
    :return:
    """

    from chest_ct_database.report_manager.establish_report_dict import get_inherent_noise

    def func_update_report(key_name, list_reference_dict, report_dict=None):

        rescaled_ct_path = os.path.join(list_reference_dict[0], key_name + '.npz')
        blood_vessel_mask_path = os.path.join(list_reference_dict[1], key_name + '.npz')

        rescaled_ct = np.load(rescaled_ct_path)['array']
        blood_vessel_mask = np.load(blood_vessel_mask_path)['array']

        case_dict = get_inherent_noise(rescaled_ct, blood_vessel_mask, is_depth_mask=False, denoise=denoise,
                                       show=False, version=version)

        return case_dict

    def func_check_processed(report_dict, key_name):

        case_dict = report_dict[key_name]

        report_keys = list(case_dict.keys())

        if denoise is True:
            if version is None:
                if 'inherent_noise (denoise)' in report_keys and 'noise_std (denoise)' in report_keys and \
                        'sampling_points_for_noise (denoise)' in report_keys and \
                        'average_blood HU (denoise)' in report_keys:
                    return True
                return False
            else:
                if "inherent_noise (denoise " + version + ")" in report_keys and \
                        "noise_std (denoise " + version + ")" in report_keys and \
                        "sampling_points_for_noise (denoise " + version + ")" in report_keys and \
                        "average_blood HU (denoise " + version + ")" in report_keys:
                    return True
                return False
        else:
            if 'inherent_noise' in report_keys and 'noise_std' in report_keys and \
                    'sampling_points_for_noise' in report_keys and 'average_blood HU' in report_keys and \
                    'max_blood_depth' in report_keys:
                return True
            return False

    update_report_dict_one_dataset(top_dict_rescaled_ct, report_save_path, func_update_report, func_check_processed,
                                   [top_dict_rescaled_ct, os.path.join(top_dict_semantics, 'blood_mask')])


def update_inherent_noise_database(feature_top_dict_rescaled_ct, feature_top_dict_semantic, top_dict_report,
                                   denoise=True, version=None):
    """

    :param version: version for the denoise model
    :param feature_top_dict_semantic:
    :param feature_top_dict_rescaled_ct:
    :param top_dict_report:
    :param denoise:
    :return:
    """
    from chest_ct_database.report_manager.establish_report_dict import get_inherent_noise

    def func_update_report(key_name, list_reference_dict, report_dict=None):

        rescaled_ct_path = os.path.join(list_reference_dict[0], key_name + '.npz')
        blood_vessel_mask_path = os.path.join(list_reference_dict[1], 'blood_mask', key_name + '.npz')

        rescaled_ct = np.load(rescaled_ct_path)['array']
        blood_vessel_mask = np.load(blood_vessel_mask_path)['array']

        case_dict = get_inherent_noise(rescaled_ct, blood_vessel_mask, is_depth_mask=False, denoise=denoise,
                                       show=False, version=version)

        return case_dict

    def func_check_processed(report_dict, key_name):

        case_dict = report_dict[key_name]

        report_keys = list(case_dict.keys())

        if denoise is True:
            if version is None:
                if 'inherent_noise (denoise)' in report_keys and 'noise_std (denoise)' in report_keys and \
                        'sampling_points_for_noise (denoise)' in report_keys and \
                        'average_blood HU (denoise)' in report_keys:
                    return True
                return False
            else:
                if "inherent_noise (denoise " + version + ")" in report_keys and \
                        "noise_std (denoise " + version + ")" in report_keys and \
                        "sampling_points_for_noise (denoise " + version + ")" in report_keys and \
                        "average_blood HU (denoise " + version + ")" in report_keys:
                    return True
                return False
        else:
            if 'inherent_noise' in report_keys and 'noise_std' in report_keys and \
                    'sampling_points_for_noise' in report_keys and 'average_blood HU' in report_keys and \
                    'max_blood_depth' in report_keys:
                return True
            return False

    update_report_dict_database(feature_top_dict_rescaled_ct, top_dict_report, func_update_report, func_check_processed,
                                [feature_top_dict_rescaled_ct, feature_top_dict_semantic])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    update_inherent_noise_one_dataset('/data_disk/RAD-ChestCT_dataset/rescaled_ct_denoise_float16/',
                                      '/data_disk/RAD-ChestCT_dataset/semantic_in_rescaled_ct/',
                                      '/data_disk/RAD-ChestCT_dataset/report_dict.pickle',
                                      denoise=True, version='epoch_5')

    exit()
    update_inherent_noise_one_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct_denoise/',
                                      '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/basic_semantics/',
                                      '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/report_dict.pickle',
                                      denoise=True, version='epoch_5')

    exit()

    update_inherent_noise_database('/data_disk/rescaled_ct_and_semantics/denoise_new_ct_float16/',
                                   '/data_disk/rescaled_ct_and_semantics/semantics/',
                                   '/data_disk/rescaled_ct_and_semantics/reports/',
                                   denoise=True, version='epoch_5')

    exit()

    update_inherent_noise_database('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/',
                                   '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/semantics/',
                                   '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/reports/',
                                   denoise=False)

    exit()
