import format_convert.dcm_np_converter_new as dcm_to_np
import collaborators_package.denoise_chest_ct.denoise_predict as de_noising
import numpy as np
import time
import shutil
import os


def dcm_folders_to_rescaled_array(top_dict_dcm_folders, func_patient_name, save_dict_rescaled_ct,
                                  func_whether_process=None, save_func=None, upsample_model_path=None, batch_size=2,
                                  denoise=True, exclusion_func=None, fold=(0, 1)):
    """
    change the dcm files to rescaled ct
    only for single dataset, all_file ct should has same property, like healthy dataset, COVID-19 dataset.

    data will be stored as top_dict_save_rescaled_gt/patient_name.npy(z)

    :param top_dict_dcm_folders:
    :param func_patient_name: func_patient_name(dict_for_dcm_files), returns a string, like patient-id_scan-time
    :param save_dict_rescaled_ct:
    :param func_whether_process: sometimes we may only want to process some ct
            func_whether_process(top_dict_save_rescaled_gt, save_name),
            True for process to rescaled_ct, False for not process
    :param save_func: save_func(rescaled_ct, top_dict_save_rescaled_gt, save_name, original_resolution=None)

    :param upsample_model_path:
    :param batch_size: batch size for upsample, usually equals to GPU count

    :param denoise: whether denoise the rescaled ct

    :param exclusion_func: if exclusion_func(dcm_dict) is True, exclude the scan

    :param fold

    :return:
    """

    if save_func is None:
        from functools import partial
        if denoise:
            denoise_model = de_noising.load_model()
            save_func = partial(default_save_func, denoise_model=denoise_model, batch_size=batch_size, denoise=True)
        else:
            save_func = partial(default_save_func, denoise=False)

    if func_whether_process is None:
        func_whether_process = default_func_whether_processed

    list_dcm_dict = extract_directory_only_contains_files(top_dict_dcm_folders)[fold[0]:: fold[1]]

    processed_count = 0

    for dcm_dict in list_dcm_dict:
        print("processing:", dcm_dict, processed_count, '/', len(list_dcm_dict))

        if exclusion_func is not None:
            if exclusion_func(dcm_dict):
                processed_count += 1  # exclude the scan
                continue

        save_name = func_patient_name(dcm_dict)

        print("the save name is:", save_name)

        if func_whether_process(save_dict_rescaled_ct, save_name):
            print("processed")
            processed_count += 1
            continue

        rescaled_ct, original_resolution = dcm_to_np.establish_rescale_chest_ct(
            dcm_dict, checkpoint_path_upsample=upsample_model_path,
            batch_size=batch_size, return_original_resolution=True)

        save_func(rescaled_ct, save_dict_rescaled_ct, save_name, original_resolution=original_resolution)

        processed_count += 1


def extract_directory_only_contains_files(top_dict_dcm_folders, list_of_directories=None, in_recursion=False):
    """

    extract all_file directories under top_dict_dcm_folders that only contains files

    :param top_dict_dcm_folders:
    :param in_recursion:
    :param list_of_directories
    :return: list of directories, for which only contains files
    """
    # assert os.path.isdir(top_dict_dcm_folders)

    if not in_recursion:
        list_of_directories = []
    else:
        assert list_of_directories is not None

    if os.path.isfile(top_dict_dcm_folders):
        return list_of_directories

    if check_directory_only_contains_files(top_dict_dcm_folders):
        list_of_directories.append(top_dict_dcm_folders)
    else:
        list_sub_dirs = os.listdir(top_dict_dcm_folders)
        assert len(list_sub_dirs) > 0

        for sub_dir in list_sub_dirs:

            new_top_dict_source = os.path.join(top_dict_dcm_folders, sub_dir)

            extract_directory_only_contains_files(new_top_dict_source, list_of_directories, True)

    return list_of_directories


def check_directory_only_contains_files(dict_directory):
    file_name_list = os.listdir(dict_directory)
    assert len(file_name_list) > 0
    for file_name in file_name_list:
        if os.path.isdir(os.path.join(dict_directory, file_name)):
            return False
    return True


def default_save_func(rescaled_ct, save_dict_rescaled_ct, save_name, denoise_model=None, batch_size=2, denoise=True,
                      original_resolution=None):
    """
    save the rescaled ct as de-noised compressed float16
    """
    from chest_ct_database.feature_manager.save_as_float_16 import convert_rescaled_ct_to_float16

    if denoise:
        if denoise_model is None:
            denoise_model = de_noising.load_model()

        print("de-noising...")
        rescaled_ct = de_noising.denoise_rescaled_array(rescaled_ct, denoise_model, batch_size)

    rescaled_ct = convert_rescaled_ct_to_float16(rescaled_ct)

    def save_np_array(save_dict, file_name, np_array):
        if not save_dict[-1] == '/':
            save_dict = save_dict + '/'
        if not os.path.exists(save_dict):
            os.makedirs(save_dict)

        buffer_path = '/home/zhoul0a/Desktop/transfer/buffer_file_longxi/'
        buffer_path = buffer_path + str(hash(file_name)) + str(time.time()) + '.npz'

        np.savez_compressed(buffer_path, array=np_array, original_resolution=original_resolution)

        shutil.move(buffer_path, save_dict + file_name + '.npz')

    save_np_array(save_dict_rescaled_ct, save_name, rescaled_ct)


def default_func_whether_processed(save_dict_rescaled_ct, save_name):

    current_path = os.path.join(save_dict_rescaled_ct, save_name)

    if os.path.exists(current_path):
        return True
    if os.path.exists(current_path + '.npz'):
        return True
    if os.path.exists(current_path + '.npy'):
        return True

    return False


def get_save_name_single_blind_dataset(dict_dcm_files):
    spilt_list = dict_dcm_files.split('/')
    return spilt_list[-2]


def get_save_name_pe_paired_dataset(dict_dcm_files):
    """
    file path: .../patient-id/non-contrast/.dcm
    :param dict_dcm_files: .../patient-id/non-contrast
    :return: patient-id
    """
    spilt_list = dict_dcm_files.split('/')
    return spilt_list[-2]


def get_save_name_av_dataset(dict_dcm_files):
    """
    file path: .../patient-id/dcm/.dcm
    :param dict_dcm_files: .../patient-id/non-contrast
    :return: patient-id
    """
    spilt_list = dict_dcm_files.split('/')
    return spilt_list[-2]


def get_save_name_default(dict_dcm_files):
    """
    file path: .../patient-id/.dcm
    :param dict_dcm_files: .../patient-id/
    :return: patient-id
    """
    spilt_list = dict_dcm_files.split('/')
    return spilt_list[-1]


def exclusion_func_pe_paired_dataset(dict_dcm_files):
    spilt_list = dict_dcm_files.split('/')
    # if not spilt_list[-1] == 'non-contrast':
    if not spilt_list[-1] == 'CTA':
        print("the scan is:", spilt_list[-1])
        print("exclude this scan")
        return True
    return False


if __name__ == '__main__':
    dcm_folders_to_rescaled_array('/data_disk/lung_altas/inhale_exhale_pair_one_patient/dcm_files/',
                                  get_save_name_default,
                                  '/data_disk/lung_altas/inhale_exhale_pair_one_patient/rescaled_ct/',
                                  denoise=False,
                                  exclusion_func=None)
    exit()

    dcm_folders_to_rescaled_array('/data_disk/artery_vein_project/new_data/CTA/dcm_files/',
                                  get_save_name_av_dataset,
                                  '/data_disk/artery_vein_project/new_data/CTA/rescaled_ct/',
                                  denoise=False,
                                  exclusion_func=None)
    exit()

    dcm_folders_to_rescaled_array('/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/16tmh_select/',
                                  get_save_name_pe_paired_dataset,
                                  '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/CTA/rescaled_ct/',
                                  denoise=False,
                                  exclusion_func=exclusion_func_pe_paired_dataset)
    exit()

    dcm_folders_to_rescaled_array('/home/zhoul0a/Desktop/pulmonary_embolism/single_blind_dataset/dcm_folders/',
                                  get_save_name_single_blind_dataset,
                                  '/home/zhoul0a/Desktop/pulmonary_embolism/single_blind_dataset/rescaled_ct/',
                                  denoise=False)

    exit()

    dcm_folders_to_rescaled_array('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/dcm_files/140tmh/',
                                  get_save_name_single_blind_dataset,
                                  '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct/', denoise=False)

    dcm_folders_to_rescaled_array('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/dcm_files/tmh75/',
                                  get_save_name_single_blind_dataset,
                                  '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct/', denoise=False)

    exit()
    import chest_ct_database.feature_manager.add_basic_tissue_seg as tissue_seg

    dict_rescaled_ct = '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct_denoise/'
    dict_semantic_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/basic_semantics/'

    tissue_seg.rescaled_ct_to_semantic_seg(dict_rescaled_ct,
                                           dict_semantic_top_dict, artery_vein=False, batch_size=2, fold=(0, 1),
                                           load_func=tissue_seg.default_load_func)

    exit()
    list_dict = extract_directory_only_contains_files(
        '/home/zhoul0a/Desktop/pulmonary_embolism/single_blind_dataset/dcm_folders/')
    print(len(list_dict))
    for item in list_dict:
        print(item)
    exit()
