"""
form visualization for lung, airway, blood vessel (if no a-v seg) or artery and vein (have a-v seg)
"""

import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import visualization.visualize_3d.highlight_semantics as highlight
import numpy as np


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name):
    """

    get the highlighted mask

    :param list_top_dict_reference: [top_dict_semantics, top_dict_dataset_non_contrast]
    :param dataset_sub_dir:
    :param file_name:
    :return: highlighted picture
    """

    file_path_artery = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'artery_mask', file_name)
    file_path_vein = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'vein_mask', file_name)
    file_path_blood = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'blood_mask', file_name)
    file_path_airway = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'airway_mask', file_name)
    file_path_lung = os.path.join(list_top_dict_reference[0], dataset_sub_dir, 'lung_mask', file_name)

    file_path_ct = os.path.join(list_top_dict_reference[1], dataset_sub_dir, file_name)

    print("processing:", file_path_ct)
    lung_mask = np.load(file_path_lung)['array']

    mass_center_z = Functions.get_mass_center_for_binary(lung_mask, median=True)[2]
    mass_center_z = int(mass_center_z)

    lung_mask = lung_mask[:, :, mass_center_z: mass_center_z + 2]

    ct_array = np.load(file_path_ct)['array']
    airway_mask = np.load(file_path_airway)['array'][:, :, mass_center_z: mass_center_z + 2]

    ct_array = ct_array[:, :, mass_center_z: mass_center_z + 2]
    ct_array = np.clip(ct_array + 0.5, 0, 1.1)

    highlighted = highlight.highlight_mask(lung_mask, ct_array, 'Y', transparency=0.9, further_highlight=False)
    highlighted = highlight.highlight_mask(airway_mask, highlighted, 'G', transparency=0.3, further_highlight=True)

    if os.path.exists(file_path_artery) and os.path.exists(file_path_vein):
        artery_mask = np.load(file_path_artery)['array'][:, :, mass_center_z: mass_center_z + 2]
        vein_mask = np.load(file_path_vein)['array'][:, :, mass_center_z: mass_center_z + 2]
        blood_mask = np.load(file_path_blood)['array'][:, :, mass_center_z: mass_center_z + 2]
        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'P', transparency=0.6, further_highlight=True)
        highlighted = highlight.highlight_mask(artery_mask, highlighted, 'R', transparency=0.3, further_highlight=True)
        highlighted = highlight.highlight_mask(vein_mask, highlighted, 'B', transparency=0.3, further_highlight=True)
    else:
        blood_mask = np.load(file_path_blood)['array'][:, :, mass_center_z: mass_center_z + 2]
        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'R', transparency=0.3, further_highlight=True)

    return highlighted[:, :, 1]


def func_file_save(save_dict, file_name, feature_package):
    Functions.image_save(feature_package, os.path.join(save_dict, file_name[:-4] + '.png'), dpi=300)


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, file_name[:-4] + '.png')
    if os.path.exists(path_saved):
        return True
    return False


def add_visualization_basic_semantic(top_dict_rescaled_ct, top_dict_semantics, top_dict_save, fold=(0, 1)):
    add_features.func_add_feature(top_dict_rescaled_ct, [top_dict_semantics, top_dict_rescaled_ct], top_dict_save,
                                  func_file_operation,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    add_visualization_basic_semantic(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/',
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/',
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/'
        'visualization/basic_semantic_check/', fold=(0, 1))
