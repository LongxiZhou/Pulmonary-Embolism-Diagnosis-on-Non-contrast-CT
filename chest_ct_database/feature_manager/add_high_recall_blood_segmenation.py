"""
Add high recall blood vessel mask
"""
import analysis.get_surface_rim_adjacent_mean as get_surface
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import numpy as np


def get_high_recall_blood_mask(rescaled_ct_denoise, vessel_mask, artery_mask, vein_mask, heart_mask, lung_mask):

    bounding_box_heart = Functions.get_bounding_box(heart_mask, pad=1)
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounding_box_heart
    pulmonary_region = lung_mask
    pulmonary_region[x_min: x_max, y_min: y_max, z_min: z_max] = 1
    pulmonary_region = pulmonary_region + get_surface.get_surface(pulmonary_region, outer=True, strict=False)
    pulmonary_region = pulmonary_region + get_surface.get_surface(pulmonary_region, outer=True, strict=False)

    high_recall_mask = np.clip(vessel_mask + artery_mask + vein_mask, 0, 1)
    high_recall_mask = high_recall_mask * np.array(rescaled_ct_denoise > Functions.change_to_rescaled(-200), 'float32')

    high_recall_mask = high_recall_mask * pulmonary_region

    # smooth
    high_recall_mask = high_recall_mask + get_surface.get_surface(high_recall_mask, outer=True, strict=False)
    high_recall_mask = high_recall_mask - get_surface.get_surface(high_recall_mask, outer=False, strict=False)
    return high_recall_mask


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name):
    """

    get denoise rescaled ct

    :param list_top_dict_reference: [top_dict_rescaled_ct-denoise, top_dict_semantics]
    :param dataset_sub_dir:
    :param file_name:
    :return: simulated_non_contrast_rescaled, visualize_image
    """

    file_path_denoise_ct = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    dict_semantics = os.path.join(list_top_dict_reference[1], dataset_sub_dir)

    print("loading:", file_path_denoise_ct)

    if file_path_denoise_ct[-1] == 'z':
        assert file_path_denoise_ct[:-4] + '.npz' == file_path_denoise_ct
        rescaled_cta_denoise = np.load(file_path_denoise_ct)['array']
    else:
        assert file_path_denoise_ct[:-4] + '.npy' == file_path_denoise_ct
        rescaled_cta_denoise = np.load(file_path_denoise_ct)

    blood_mask = np.load(os.path.join(dict_semantics, 'blood_mask', file_name))['array']
    artery_mask = np.load(os.path.join(dict_semantics, 'artery_mask', file_name))['array']
    vein_mask = np.load(os.path.join(dict_semantics, 'vein_mask', file_name))['array']

    heart_mask = np.load(os.path.join(dict_semantics, 'heart_mask', file_name))['array']
    lung_mask = np.load(os.path.join(dict_semantics, 'lung_mask', file_name))['array']

    high_recall_blood_mask = get_high_recall_blood_mask(
        rescaled_cta_denoise, blood_mask, artery_mask, vein_mask, heart_mask, lung_mask)

    return high_recall_blood_mask


def func_file_save(save_dict, file_name, feature_package):
    high_recall_blood_mask = feature_package
    save_dict = os.path.join(save_dict, 'blood_mask_high_recall')
    Functions.save_np_array(save_dict, file_name[:-4] + '.npz',
                            high_recall_blood_mask, compress=True)


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, 'blood_mask_high_recall', file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def process_dataset(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_save, fold=(0, 1)):
    func_file_operation_semantic = func_file_operation
    add_features.func_add_feature(top_dict_rescaled_ct_denoise, [top_dict_rescaled_ct_denoise, top_dict_semantics],
                                  top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    current_fold = (0, 6)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)

    process_dataset(
        '/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
        '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/',
        '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/',
        fold=current_fold)
