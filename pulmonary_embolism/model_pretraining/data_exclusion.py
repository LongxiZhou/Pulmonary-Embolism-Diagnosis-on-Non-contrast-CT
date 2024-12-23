import Tool_Functions.Functions as Functions
import os
import numpy as np
import shutil


def establish_report_dict(top_dict_semantic, dict_center_line_blood, report_save_path):
    """

    :param top_dict_semantic:
    :param dict_center_line_blood:
    :param report_save_path:
    :return:
    """
    import analysis.get_surface_rim_adjacent_mean as get_surface
    import analysis.center_line_and_depth_3D as get_center_line

    file_name_list = os.listdir(dict_center_line_blood)

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

        airway_mask = np.load(top_dict_semantic + 'airway_mask/' + file_name[:-4] + '.npz')['array']
        heart_mask = np.load(top_dict_semantic + 'heart_mask/' + file_name[:-4] + '.npz')['array']
        lung_mask = np.load(top_dict_semantic + 'lung_mask/' + file_name[:-4] + '.npz')['array']
        vessel_mask = np.load(top_dict_semantic + 'blood_mask/' + file_name[:-4] + '.npz')['array']

        # get information for exclusion
        infection_mask = np.load(top_dict_semantic + 'infection/' + file_name[:-4] + '.npz')['array']

        lung_volume = np.sum(lung_mask)
        lung_surface_area = np.sum(get_surface.get_surface(lung_mask))

        heart_volume = np.sum(heart_mask)
        heart_surface_area = np.sum(get_surface.get_surface(heart_mask))

        airway_center_line_voxel = None
        try:
            airway_center_line_voxel = np.sum(get_center_line.get_center_line(airway_mask))
        except:
            print("cannot get airway center line")

        airway_volume = np.sum(airway_mask)
        airway_surface_area = np.sum(get_surface.get_surface(airway_mask))

        blood_vessel_center_line = np.load(os.path.join(dict_center_line_blood, file_name))['array']
        blood_vessel_center_line_voxel = np.sum(blood_vessel_center_line)
        blood_vessel_volume = np.sum(vessel_mask)
        blood_vessel_surface_area = np.sum(get_surface.get_surface(vessel_mask))

        infection_volume = np.sum(infection_mask)
        infection_to_lung_ratio = infection_volume / lung_volume

        case_dict = {'lung_volume': lung_volume, 'lung_surface_area': lung_surface_area, 'heart_volume': heart_volume,
                     'heart_surface_area': heart_surface_area, 'airway_center_line_voxel': airway_center_line_voxel,
                     'airway_volume': airway_volume, 'airway_surface_area': airway_surface_area,
                     'blood_vessel_center_line_voxel': blood_vessel_center_line_voxel,
                     'blood_vessel_volume': blood_vessel_volume,
                     'blood_vessel_surface_area': blood_vessel_surface_area,
                     'infection_volume': infection_volume, 'infection_to_lung_ratio': infection_to_lung_ratio}

        print("case_dict")
        print(case_dict)

        report_dict[file_name[:-4]] = case_dict

        Functions.pickle_save_object(report_save_path, report_dict)

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

    establish_report_dict('/media/zhoul0a/New Volume/RAD-ChestCT_dataset/semantic_in_rescaled_ct/',
                          '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/depth_and_center-line/center_line_mask/',
                          '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/semantic_report_dict.pickle')

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
