"""
each dataset has a report dict, with key patient-name
report dict is saved with name "report_dict.pickle"
"""


import Tool_Functions.Functions as Functions
import os
from chest_ct_database.basic_functions import extract_relative_dirs_sub_dataset, merge_dicts
import numpy as np
import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.center_line_and_depth_3D as get_center_line
import shutil
import warnings


def get_case_dict_basic_semantic(lung_mask, heart_mask, blood_vessel_mask, airway_mask, infection_mask,
                                 airway_center_line_mask=None, blood_vessel_center_line_mask=None, show=True):

    lung_volume = np.sum(lung_mask)
    lung_surface_area = np.sum(get_surface.get_surface(lung_mask))

    heart_volume = np.sum(heart_mask)
    heart_surface_area = np.sum(get_surface.get_surface(heart_mask))

    if airway_center_line_mask is None:
        airway_center_line_mask = get_center_line.get_center_line(airway_mask)

    airway_center_line_voxel = np.sum(airway_center_line_mask)

    airway_volume = np.sum(airway_mask)
    airway_surface_area = np.sum(get_surface.get_surface(airway_mask))

    if blood_vessel_center_line_mask is None:
        blood_vessel_center_line_mask = get_center_line.get_center_line(blood_vessel_mask)
    blood_vessel_center_line_voxel = np.sum(blood_vessel_center_line_mask)
    blood_vessel_volume = np.sum(blood_vessel_mask)
    blood_vessel_surface_area = np.sum(get_surface.get_surface(blood_vessel_mask))

    infection_volume = np.sum(infection_mask)
    infection_to_lung_ratio = infection_volume / lung_volume

    case_dict = {'lung_volume': lung_volume, 'lung_surface_area': lung_surface_area, 'heart_volume': heart_volume,
                 'heart_surface_area': heart_surface_area, 'airway_center_line_voxel': airway_center_line_voxel,
                 'airway_volume': airway_volume, 'airway_surface_area': airway_surface_area,
                 'blood_vessel_center_line_voxel': blood_vessel_center_line_voxel,
                 'blood_vessel_volume': blood_vessel_volume,
                 'blood_vessel_surface_area': blood_vessel_surface_area,
                 'infection_volume': infection_volume, 'infection_to_lung_ratio': infection_to_lung_ratio}

    if show:
        print(case_dict)

    return case_dict


def get_inherent_noise(rescaled_ct, blood_vessel_depth, is_depth_mask=False, show=True, depth_difference=5,
                       denoise=True, version=None):
    """

    :param version: record denoise model version
    :param rescaled_ct:
    :param blood_vessel_depth: can provide blood vessel mask or blood vessel encoding_depth mask
    :param is_depth_mask:
    :param show
    :param depth_difference
    :param denoise: whether the rescaled_ct is denoise
    :return: case dict for inherent noise
    """
    if not is_depth_mask:
        blood_vessel_depth = get_center_line.get_surface_distance(blood_vessel_depth)

    max_depth = np.max(blood_vessel_depth)

    if not max_depth > depth_difference > 0:
        warnings.warn("seems not chest scan.")
        if max_depth > 1:
            depth_difference = 1
        else:
            raise ValueError("wrong scan or wrong seg")

    if max_depth < 20:
        print("max encoding_depth is:", max_depth)
        warnings.warn("Value Warning: the max encoding_depth of the blood vessel is too small.")

    if show:
        print("max encoding_depth is:", max_depth)

    mask_sampling = np.array(blood_vessel_depth > (max_depth - depth_difference), 'int16')
    non_zero_loc = Functions.get_location_list(np.where(mask_sampling > 0.5))

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

    if version is not None:
        assert denoise is True

        case_dict = {"inherent_noise (denoise " + version + ")": abs_differ_ave,
                     "noise_std (denoise " + version + ")": std,
                     "sampling_points_for_noise (denoise " + version + ")": len(ct_value),
                     "average_blood HU (denoise " + version + ")": mean}

    else:
        if denoise:
            case_dict = {"inherent_noise (denoise)": abs_differ_ave, "noise_std (denoise)": std,
                         "sampling_points_for_noise (denoise)": len(ct_value), "average_blood HU (denoise)": mean}
        else:
            case_dict = {"inherent_noise": abs_differ_ave, "noise_std": std,
                         "sampling_points_for_noise": len(ct_value), "average_blood HU": mean,
                         "max_blood_depth": max_depth}

    if show:
        print(case_dict)

    return case_dict


def establish_report_dict_one_dataset(top_dict_semantic, top_dict_rescaled_ct, report_save_path,
                                      dict_center_line_blood=None, dict_center_line_airway=None, denoise=True):
    """

    require to first get the semantic segmentation: blood vessel, lungs, heart, infection and airways

    the save name, or keys in the report dict is like patient-id_time

    :param top_dict_semantic:
    :param top_dict_rescaled_ct: saved in .npz
    :param dict_center_line_blood:
    :param report_save_path:
    :param dict_center_line_airway
    :param denoise: whether the rescaled ct is after denoise
    :return:
    """
    if dict_center_line_blood is not None:
        if not os.path.exists(dict_center_line_blood):
            print("dict_center_line_blood:", dict_center_line_blood)
            warnings.warn("dict_center_line_blood not exist")
            dict_center_line_blood = None
    if dict_center_line_airway is not None:
        if not os.path.exists(dict_center_line_airway):
            print("dict_center_line_airway:", dict_center_line_airway)
            warnings.warn("dict_center_line_airway not exist")
            dict_center_line_airway = None

    file_name_list = os.listdir(os.path.join(top_dict_semantic, 'blood_mask'))

    file_name_list_rescaled_ct = os.listdir(top_dict_rescaled_ct)

    if os.path.exists(report_save_path):
        report_dict = Functions.pickle_load_object(report_save_path)
    else:
        report_dict = {}

    processed_list = list(report_dict.keys())
    processed_count = 0

    for file_name in file_name_list:

        print("processing:", file_name[:-4], processed_count, '/', len(file_name_list))

        if file_name[:-4] in processed_list:
            print("processed")
            processed_count += 1
            continue
        if file_name not in file_name_list_rescaled_ct and file_name[:-4] + '.npy' not in file_name_list_rescaled_ct\
                and file_name[:-4] + '_unknown.npz' not in file_name_list_rescaled_ct:
            print("file name:", file_name, "not has rescaled ct at path:")
            print(os.path.join(top_dict_rescaled_ct, file_name))
            warnings.warn("this file name not has rescale ct")
            processed_count += 1
            continue

        if os.path.exists(os.path.join(top_dict_semantic, 'heart_mask', file_name)):
            heart_mask = np.load(os.path.join(top_dict_semantic, 'heart_mask', file_name))['array']
            heart_volume = np.sum(heart_mask)
            heart_surface_area = np.sum(get_surface.get_surface(heart_mask))
        else:
            heart_volume = None
            heart_surface_area = None
        if os.path.exists(os.path.join(top_dict_semantic, 'lung_mask', file_name)):
            lung_mask = np.load(os.path.join(top_dict_semantic, 'lung_mask', file_name))['array']
            lung_volume = np.sum(lung_mask)
            lung_surface_area = np.sum(get_surface.get_surface(lung_mask))
        else:
            lung_volume = None
            lung_surface_area = None

        blood_vessel_mask = np.load(os.path.join(top_dict_semantic, 'blood_mask', file_name))['array']
        airway_mask = np.load(os.path.join(top_dict_semantic, 'airway_mask', file_name))['array']

        if os.path.exists(os.path.join(top_dict_rescaled_ct, file_name)):
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, file_name))['array']
        else:
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, file_name[:-4] + '_unknown.npz'))['array']

        if os.path.exists(os.path.join(top_dict_semantic, 'infection', file_name)):
            infection_mask = np.load(os.path.join(top_dict_semantic, 'infection', file_name))['array']
            infection_volume = np.sum(infection_mask)
            if lung_volume is not None:
                infection_to_lung_ratio = infection_volume / lung_volume
            else:
                infection_to_lung_ratio = None
        else:
            infection_volume = None
            infection_to_lung_ratio = None

        if dict_center_line_airway is not None:
            if os.path.exists(os.path.join(dict_center_line_airway, file_name)):
                airway_center_line = np.load(os.path.join(dict_center_line_airway, file_name))['array']
            else:
                airway_center_line = get_center_line.get_center_line(airway_mask)
                Functions.save_np_array(dict_center_line_airway, file_name, airway_center_line, compress=True)
        else:
            airway_center_line = get_center_line.get_center_line(airway_mask)
        airway_center_line_voxel = np.sum(airway_center_line)

        airway_volume = np.sum(airway_mask)
        airway_surface_area = np.sum(get_surface.get_surface(airway_mask))

        if dict_center_line_blood is not None:
            if os.path.exists(os.path.join(dict_center_line_blood, file_name)):
                blood_vessel_center_line = np.load(os.path.join(dict_center_line_blood, file_name))['array']
            else:
                blood_vessel_center_line = get_center_line.get_center_line(blood_vessel_mask)
                Functions.save_np_array(dict_center_line_blood, file_name, blood_vessel_center_line, compress=True)
        else:
            blood_vessel_center_line = get_center_line.get_center_line(blood_vessel_mask)

        blood_vessel_center_line_voxel = np.sum(blood_vessel_center_line)
        blood_vessel_volume = np.sum(blood_vessel_mask)
        blood_vessel_surface_area = np.sum(get_surface.get_surface(blood_vessel_mask))

        case_dict = {'lung_volume': lung_volume, 'lung_surface_area': lung_surface_area, 'heart_volume': heart_volume,
                     'heart_surface_area': heart_surface_area, 'airway_center_line_voxel': airway_center_line_voxel,
                     'airway_volume': airway_volume, 'airway_surface_area': airway_surface_area,
                     'blood_vessel_center_line_voxel': blood_vessel_center_line_voxel,
                     'blood_vessel_volume': blood_vessel_volume,
                     'blood_vessel_surface_area': blood_vessel_surface_area,
                     'infection_volume': infection_volume, 'infection_to_lung_ratio': infection_to_lung_ratio}

        inherent_noise_dict = get_inherent_noise(rescaled_ct, blood_vessel_mask, is_depth_mask=False, show=False,
                                                 denoise=denoise)

        case_dict = merge_dicts([case_dict, inherent_noise_dict])

        print("case_dict")
        print(case_dict)

        report_dict[file_name[:-4]] = case_dict

        Functions.pickle_save_object(report_save_path, report_dict)

        processed_count += 1


def establish_report_dict_database(top_dict_rescaled_ct, denoise=True):
    """

    :param top_dict_rescaled_ct: in .npz, and can serve as the source to extract sub_dirs for dataset
    :param denoise: whether the top_dict_for_file_names is after denoise
    :return:
    """
    list_sub_dataset = extract_relative_dirs_sub_dataset(top_dict_rescaled_ct)

    database_dict = Functions.get_father_dict(top_dict_rescaled_ct)

    processed_count = 0
    for sub_dataset in list_sub_dataset:
        print("\n#########################################")
        print("establishing report dict for", sub_dataset, processed_count, '/', len(list_sub_dataset))
        print("#########################################")

        top_dict_semantic = os.path.join(database_dict, 'semantics', sub_dataset)

        report_save_path = os.path.join(database_dict, 'reports', sub_dataset, 'report_dict.pickle')

        dict_center_line_airway = os.path.join(
            database_dict, 'depth_and_center-line', sub_dataset, 'airway_center_line')

        dict_center_line_blood = os.path.join(
            database_dict, 'depth_and_center-line', sub_dataset, 'blood_center_line')

        top_dict_rescaled_ct_dataset = os.path.join(top_dict_rescaled_ct, sub_dataset)

        establish_report_dict_one_dataset(top_dict_semantic, top_dict_rescaled_ct_dataset, report_save_path,
                                          dict_center_line_blood, dict_center_line_airway, denoise=denoise)

        processed_count += 1


def exclusion_with_lesion_ratio_and_center_line(dict_report_path, image_check_dict, save_image_dict):
    """
    require the 'infection_to_lung_ratio' < 0.075 and 'blood_vessel_center_line_voxel' > 6000
    :param dict_report_path:
    :param image_check_dict:
    :param save_image_dict: we paste the check image to the save_image_path
    :return: None
    """

    dict_report = Functions.pickle_load_object(dict_report_path)
    print("there are:", len(dict_report), 'scans')

    if not os.path.exists(save_image_dict):
        os.makedirs(save_image_dict)

    for key in dict_report.keys():
        if dict_report[key]['infection_to_lung_ratio'] < 0.075 and \
                dict_report[key]['blood_vessel_center_line_voxel'] > 6000:
            shutil.copyfile(os.path.join(image_check_dict, key[:-4] + '.png'),
                            os.path.join(save_image_dict, key[:-4] + '.png'))


def establish_merged(dataset_dict_list, good_quality_dict_list, dict_merge):
    """

    :param dataset_dict_list: list of directory for pickle dataset
    :param good_quality_dict_list: stores png for good quality scans. list of directory for scan_names of inclusion
    :param dict_merge: the dict to restore merged pickle dataset
    :return: None
    """
    assert len(dataset_dict_list) == len(good_quality_dict_list)

    for dataset_id in range(len(dataset_dict_list)):
        dataset_pickle_dict = dataset_dict_list[dataset_id]
        name_list_inclusion = os.listdir(good_quality_dict_list[dataset_id])
        for name in name_list_inclusion:
            pickle_path = os.path.join(dataset_pickle_dict, name[:-4] + '.pickle')
            save_path = os.path.join(dict_merge, name[:-4] + '.pickle')
            shutil.copyfile(pickle_path, save_path)
    return None


if __name__ == '__main__':

    establish_report_dict_one_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/semantics/',
                                      '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/denoise-rescaled_ct/',
                                      '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/report_dict.pickle',
                                      denoise=True, dict_center_line_blood='/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/depth_and_center-line/blood_center_line/',
                                      dict_center_line_airway='/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/depth_and_center-line/airway_center_line/')
    exit()

    establish_report_dict_database('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/denoise_ct_float16/',
                                   denoise=True)

    exit()
    vessel_dataset_dict_list = \
        ['/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_healthy/blood_vessels/',
         '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_mudanjiang/blood_vessels/',
         '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_yidayi/blood_vessels/']
    vessel_good_quality_dict_list = \
        ['/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_healthy/good_quality/',
         '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_mudanjiang/image_good_quality/',
         '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_yidayi/image_good_quality/']

    vessel_merge_dataset_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/blood_vessel_merge/'

    establish_merged(vessel_dataset_dict_list, vessel_good_quality_dict_list, vessel_merge_dataset_dict)
    exit()
    exclusion_with_lesion_ratio_and_center_line(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_yidayi/dict_report.pickle',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_yidayi/image_check/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_yidayi/image_good_quality/')
    exit()
    exclusion_with_lesion_ratio_and_center_line(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_mudanjiang/dict_report.pickle',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_mudanjiang/image_check/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset_mudanjiang/image_good_quality/')
