"""
form visualization for blood region (red) and blood region strict (blue)
"""

import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import visualization.visualize_3d.highlight_semantics as highlight
import numpy as np


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name):
    """

    get the highlighted mask

    :param list_top_dict_reference: [top_dict_secondary_semantics, top_dict_rescaled_ct]
    :param dataset_sub_dir:
    :param file_name:
    :return: highlighted picture
    """

    file_path_blood_region = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'blood_region', file_name)
    file_path_blood_region_strict = os.path.join(
        list_top_dict_reference[0], dataset_sub_dir, 'blood_region_strict', file_name)

    file_path_ct = os.path.join(list_top_dict_reference[1], dataset_sub_dir, file_name)

    print("processing:", file_path_ct)
    ct_array = np.load(file_path_ct)['array']

    blood_region = np.load(file_path_blood_region)['array'][:, :, 255: 257]
    blood_region[0: 20, 0: 40] = 1  # illustrating the color for blood region
    blood_region_strict = np.load(file_path_blood_region_strict)['array'][:, :, 255: 257]
    blood_region_strict[0: 20, 20: 60] = 1  # illustrating the color for blood region_strict

    ct_array = ct_array[:, :, 255: 257]
    ct_array[0: 20, 0: 60] = 0.5
    ct_array = np.clip(ct_array + 0.25, 0, 0.8125)  # [-1000 HU, 300 HU]

    highlighted = highlight.highlight_mask(blood_region, ct_array, 'R', transparency=0.5, further_highlight=False)
    highlighted = highlight.highlight_mask(blood_region_strict, highlighted, 'B',
                                           transparency=0.5, further_highlight=True)

    return highlighted[:, :, 1]


def func_file_save(save_dict, file_name, feature_package):
    Functions.image_save(feature_package, os.path.join(save_dict, file_name[:-4] + '.png'), dpi=300)


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, file_name[:-4] + '.png')
    if os.path.exists(path_saved):
        return True
    return False


def add_visualization_blood_region(top_dict_rescaled_ct, top_dict_semantics, top_dict_save, fold=(0, 1)):
    add_features.func_add_feature(top_dict_rescaled_ct, [top_dict_semantics, top_dict_rescaled_ct], top_dict_save,
                                  func_file_operation,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    add_visualization_blood_region(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/',
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/secondary_semantics/',
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/visualization/blood_region_check/',
        fold=(0, 2))
