
"""
mha is the ground truth for certain dcm_files.

As dcm_files follows the architecture:
patient-id/.../dcm_files/

mha files must have the following architecture:
patient-id/.../semantic-name.mha


"""

from chest_ct_database.initialize_or_pipeline_process.pipeline_dcm_to_npz import extract_directory_only_contains_files
import format_convert.dcm_np_converter_new as mha_to_npz
import Tool_Functions.Functions as Functions
import numpy as np
import visualization.visualize_3d.highlight_semantics as highlight
import os


def mha_file_to_rescaled_gt(top_dict_dcm_folders, func_patient_name, top_dict_save_rescaled_gt,
                            func_whether_process, save_func, exclusion_func=None, fold=(0, 1)):
    """
    change the mha files or npz to rescaled gt
    only for single dataset, all_file ct should has same property, like healthy dataset, COVID-19 dataset.

    data will be stored as top_dict_save_rescaled_gt/semantic-name/patient-name.npz

    :param fold:
    :param top_dict_dcm_folders:
    :param func_patient_name: func_patient_name(dict_for_dcm_files), returns a string, like patient-id_scan-time
            get the same_name
    :param top_dict_save_rescaled_gt:
    :param func_whether_process: sometimes we may only want to process some ct
            func_whether_process(top_dict_save_rescaled_gt, save_name),
            True for processed, False for not process
    :param save_func: save_func(dcm_dict, top_dict_save_rescaled_gt, save_name, original_resolution=None)

    :param exclusion_func: if exclusion_func(dcm_dict) is True, exclude the scan

    :return:
    """

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

        if func_whether_process(top_dict_save_rescaled_gt, save_name):
            print("processed")
            processed_count += 1
            continue

        save_func(dcm_dict, top_dict_save_rescaled_gt, save_name)

        processed_count += 1


def save_func_av_an_dataset(dcm_dict, top_dict_save_rescaled_gt, save_name, original_resolution=None):
    """

     on the same level for every dcm_dict, contains 4 .mha files.

    :param dcm_dict:
    :param top_dict_save_rescaled_gt:
    :param save_name:
    :param original_resolution:
    :return:
    """
    if original_resolution is None:
        original_resolution = mha_to_npz.get_resolution_from_dcm(dcm_dict, show=True)

    father_dict = Functions.get_father_dict(dcm_dict)

    artery_mha_path = os.path.join(father_dict, 'atery.mha')
    vein_mha_path = os.path.join(father_dict, 'vein.mha')
    airway_mha_path = os.path.join(father_dict, 'airway.mha')
    nodule_mha_path = os.path.join(father_dict, 'nodule.mha')

    artery_gt = mha_to_npz.establish_rescaled_mask(artery_mha_path, resolutions=original_resolution)
    vein_gt = mha_to_npz.establish_rescaled_mask(vein_mha_path, resolutions=original_resolution)
    airway_gt = mha_to_npz.establish_rescaled_mask(airway_mha_path, resolutions=original_resolution)
    if os.path.exists(nodule_mha_path):
        nodule_gt = mha_to_npz.establish_rescaled_mask(nodule_mha_path, resolutions=original_resolution)
    else:
        nodule_gt = np.zeros([512, 512, 512], 'float16')

    save_name = save_name + '.npz'

    Functions.save_np_array(os.path.join(top_dict_save_rescaled_gt, 'artery_gt'), save_name, artery_gt, compress=True)
    Functions.save_np_array(os.path.join(top_dict_save_rescaled_gt, 'vein_gt'), save_name, vein_gt, compress=True)
    Functions.save_np_array(os.path.join(top_dict_save_rescaled_gt, 'airway_gt'), save_name, airway_gt, compress=True)
    Functions.save_np_array(os.path.join(top_dict_save_rescaled_gt, 'nodule_gt'), save_name, nodule_gt, compress=True)


def whether_processed_av_an_dataset(top_dict_save_rescaled_gt, save_name):
    processed_artery = os.path.exists(os.path.join(top_dict_save_rescaled_gt, 'artery_gt', save_name + '.npz'))
    processed_vein = os.path.exists(os.path.join(top_dict_save_rescaled_gt, 'vein_gt', save_name + '.npz'))
    processed_airway = os.path.exists(os.path.join(top_dict_save_rescaled_gt, 'airway_gt', save_name + '.npz'))
    processed_nodule = os.path.exists(os.path.join(top_dict_save_rescaled_gt, 'nodule_gt', save_name + '.npz'))

    if processed_artery and processed_vein and processed_airway and processed_nodule:
        return True
    return False


def get_save_name_av_dataset(dict_dcm_files):
    """
    file path: .../patient-id/dcm/.dcm
    :param dict_dcm_files: .../patient-id/non-contrast
    :return:
    """
    spilt_list = dict_dcm_files.split('/')
    return spilt_list[-2]


def pipeline_check_av_an_dataset(top_dict_rescaled_ct, top_dict_gt, top_dict_save, fold=(0, 1)):

    file_name_list = os.listdir(top_dict_rescaled_ct)[fold[0]::fold[1]]

    def check_processed(file_name):
        save_path_middle = os.path.join(top_dict_save, file_name[:-4] + '_middle.png')
        save_path_nodule = os.path.join(top_dict_save, file_name[:-4] + '_nodule.png')
        processed_middle = os.path.exists(save_path_middle)
        processed_nodule = os.path.exists(save_path_nodule)
        if processed_middle and processed_nodule:
            return True
        return False

    def process_one(file_name):
        print(file_name)

        rescaled_ct = np.load(top_dict_rescaled_ct + file_name)['array']
        rescaled_ct = np.clip(rescaled_ct + 0.5, 0, 1)

        artery = np.load(top_dict_gt + 'artery_gt/' + file_name)['array']
        vein = np.load(top_dict_gt + 'vein_gt/' + file_name)['array']
        airway = np.load(top_dict_gt + 'airway_gt/' + file_name)['array']
        if os.path.exists(top_dict_gt + 'nodule_gt/' + file_name):
            nodule = np.load(top_dict_gt + 'nodule_gt/' + file_name)['array']
        else:
            nodule = np.zeros([512, 512, 512], 'float16')
        nodule_loc_z = np.where(nodule > 0.5)[2]
        if len(nodule_loc_z) > 0:
            nodule_center = int(np.median(nodule_loc_z))
        else:
            nodule_center = 0

        highlighted = highlight.highlight_mask(
            artery, rescaled_ct, channel='R', further_highlight=False, transparency=0.3)
        highlighted = highlight.highlight_mask(
            vein, highlighted, channel='B', further_highlight=True, transparency=0.3)
        highlighted = highlight.highlight_mask(
            airway, highlighted, channel='G', further_highlight=True, transparency=0.3)
        highlighted = highlight.highlight_mask(
            nodule, highlighted, channel='Y', further_highlight=True, transparency=0.3)

        save_path_middle = os.path.join(top_dict_save, file_name[:-4] + '_middle.png')
        save_path_nodule = os.path.join(top_dict_save, file_name[:-4] + '_nodule.png')
        Functions.image_save(highlighted[:, :, 256], save_path_middle, dpi=300)
        if nodule_center > 0:
            Functions.image_save(highlighted[:, :, nodule_center], save_path_nodule, dpi=300)

    processed = 0
    for fn in file_name_list:
        print("processing:", fn, processed, '/', len(file_name_list))
        if check_processed(fn):
            print("processed")
            processed += 1
            continue
        process_one(fn)
        processed += 1


if __name__ == '__main__':
    mha_file_to_rescaled_gt('/data_disk/artery_vein_project/new_data/CTA/dcm_files',
                            get_save_name_av_dataset,
                            '/data_disk/artery_vein_project/new_data/CTA/ground_truth',
                            whether_processed_av_an_dataset,
                            save_func_av_an_dataset, exclusion_func=None, fold=(0, 2))

    pipeline_check_av_an_dataset('/data_disk/artery_vein_project/new_data/CTA/rescaled_ct/',
                                 '/data_disk/artery_vein_project/new_data/CTA/ground_truth/',
                                 '/data_disk/artery_vein_project/new_data/CTA/visualization/check_gt/',
                                 fold=(0, 2))
    exit()
