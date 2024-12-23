"""
form visualization for lung, airway, blood vessel (if no a-v seg) or artery and vein (have a-v seg)
"""

import Tool_Functions.Functions as Functions
import os
import visualization.visualize_3d.highlight_semantics as highlight
from pe_dataset_management.registration.register_ct_to_cta.inferenece_fast_ct_to_cta import cast_non_contrast_to_cta
from pe_dataset_management.basic_functions import find_patient_id_dataset_correspondence, get_all_scan_name
import numpy as np


def images_not_register(list_top_dict_reference, scan_name):
    """

    get the highlighted mask

    :param list_top_dict_reference: [dict_semantics, dict_rescaled_ct]
    :param scan_name:
    :return: highlighted picture
    """

    file_path_artery = os.path.join(list_top_dict_reference[0], 'artery_mask', scan_name)
    file_path_vein = os.path.join(list_top_dict_reference[0], 'vein_mask', scan_name)
    file_path_blood = os.path.join(list_top_dict_reference[0], 'blood_mask', scan_name)
    file_path_airway = os.path.join(list_top_dict_reference[0], 'airway_mask', scan_name)
    file_path_lung = os.path.join(list_top_dict_reference[0], 'lung_mask', scan_name)

    file_path_ct = os.path.join(list_top_dict_reference[1], scan_name)

    print("processing:", file_path_ct)
    ct_array = np.load(file_path_ct)['array']
    airway_mask = np.load(file_path_airway)['array']
    lung_mask = np.load(file_path_lung)['array']

    ct_array = ct_array
    ct_array = np.clip(ct_array + 0.5, 0, 1.2)

    highlighted = highlight.highlight_mask(lung_mask, ct_array, 'Y', transparency=0.9, further_highlight=False)
    highlighted = highlight.highlight_mask(airway_mask, highlighted, 'G', transparency=0.3, further_highlight=True)

    if os.path.exists(file_path_artery) and os.path.exists(file_path_vein):
        artery_mask = np.load(file_path_artery)['array']
        vein_mask = np.load(file_path_vein)['array']
        blood_mask = np.load(file_path_blood)['array']
        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'P', transparency=0.6, further_highlight=True)
        highlighted = highlight.highlight_mask(artery_mask, highlighted, 'R', transparency=0.3, further_highlight=True)
        highlighted = highlight.highlight_mask(vein_mask, highlighted, 'B', transparency=0.3, further_highlight=True)
    else:
        blood_mask = np.load(file_path_blood)['array']
        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'R', transparency=0.3, further_highlight=True)

    mass_center = Functions.get_mass_center_for_binary(lung_mask)

    image_z = highlighted[:, :, int(mass_center[2])]
    image_x = highlighted[int(mass_center[0]), :, :]
    image_x = np.swapaxes(image_x, 0, 1)
    image_x = np.flip(image_x, axis=0)
    image_y = np.zeros([512, 1024, 3], 'float32')
    image_y[:, 0: 512, :] = highlighted[:, int(mass_center[1]), :]
    image_y[:, 512:, :] = highlighted[:, int(mass_center[1]) + 512, :]

    return image_x, image_y, image_z, mass_center


def images_non_contrast_registered(list_top_dict_reference, scan_name, mass_center=None):
    """

    get the highlighted mask

    :param mass_center:
    :param list_top_dict_reference: [dict_semantics, dict_rescaled_ct]
    :param scan_name:
    :return: highlighted picture
    """

    file_path_artery = os.path.join(list_top_dict_reference[0], 'artery_mask', scan_name)
    file_path_vein = os.path.join(list_top_dict_reference[0], 'vein_mask', scan_name)
    file_path_blood = os.path.join(list_top_dict_reference[0], 'blood_mask', scan_name)
    file_path_airway = os.path.join(list_top_dict_reference[0], 'airway_mask', scan_name)
    file_path_lung = os.path.join(list_top_dict_reference[0], 'lung_mask', scan_name)

    file_path_ct = os.path.join(list_top_dict_reference[1], scan_name)

    print("processing:", file_path_ct)
    ct_array = np.load(file_path_ct)['array']
    airway_mask = np.load(file_path_airway)['array']
    lung_mask = np.load(file_path_lung)['array']

    ct_array = ct_array
    ct_array = np.clip(ct_array + 0.5, 0, 1.2)

    # registration
    ct_array, airway_mask, lung_mask = cast_non_contrast_to_cta([ct_array, airway_mask, lung_mask], scan_name)

    highlighted = highlight.highlight_mask(lung_mask, ct_array, 'Y', transparency=0.9, further_highlight=False)
    highlighted = highlight.highlight_mask(airway_mask, highlighted, 'G', transparency=0.3, further_highlight=True)

    if os.path.exists(file_path_artery) and os.path.exists(file_path_vein):
        artery_mask = np.load(file_path_artery)['array']
        vein_mask = np.load(file_path_vein)['array']
        blood_mask = np.load(file_path_blood)['array']

        # registration
        artery_mask, vein_mask, blood_mask = cast_non_contrast_to_cta([artery_mask, vein_mask, blood_mask], scan_name)

        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'P', transparency=0.6, further_highlight=True)
        highlighted = highlight.highlight_mask(artery_mask, highlighted, 'R', transparency=0.3, further_highlight=True)
        highlighted = highlight.highlight_mask(vein_mask, highlighted, 'B', transparency=0.3, further_highlight=True)
    else:
        blood_mask = np.load(file_path_blood)['array']

        # registration
        blood_mask = cast_non_contrast_to_cta(blood_mask, scan_name)

        highlighted = highlight.highlight_mask(blood_mask, highlighted, 'R', transparency=0.3, further_highlight=True)

    if mass_center is None:
        mass_center = Functions.get_mass_center_for_binary(lung_mask)

    image_z = highlighted[:, :, int(mass_center[2])]
    image_x = highlighted[int(mass_center[0]), :, :]
    image_x = np.swapaxes(image_x, 0, 1)
    image_x = np.flip(image_x, axis=0)
    image_y = np.zeros([512, 1024, 3], 'float32')
    image_y[:, 0: 512, :] = highlighted[:, int(mass_center[1]), :]
    image_y[:, 512:, :] = highlighted[:, int(mass_center[1]) + 512, :]

    return image_x, image_y, image_z, mass_center


def process_scan(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset'):
    if scan_name is not None:
        if len(scan_name) <= 4:
            scan_name = scan_name + '.npz'
        if len(scan_name) > 4:
            if not scan_name[-4:] == '.npz':
                scan_name = scan_name + '.npz'
    dataset_dict_cta, dataset_dict_non_contrast = \
        find_patient_id_dataset_correspondence(top_dict=top_dict, scan_name=scan_name, check_pair=False)[scan_name[:-4]]

    dict_semantics_non = os.path.join(dataset_dict_non_contrast, 'semantics')
    dict_rescaled_ct_non = os.path.join(dataset_dict_non_contrast, 'rescaled_ct-denoise')

    dict_semantics_cta = os.path.join(dataset_dict_cta, 'semantics')
    dict_rescaled_cta = os.path.join(dataset_dict_cta, 'rescaled_ct-denoise')

    image_x, image_y, image_z, mass_center = images_not_register([dict_semantics_cta, dict_rescaled_cta], scan_name)

    image_x_r, image_y_r, image_z_r, mass_center = images_non_contrast_registered(
        [dict_semantics_non, dict_rescaled_ct_non], scan_name, mass_center=mass_center)

    image_x_c, image_y_c, image_z_c, mass_center = images_not_register(
        [dict_semantics_non, dict_rescaled_ct_non], scan_name)

    image_x_combined = np.concatenate((image_x, image_x_r, image_x_c), axis=0)
    image_y_combined = np.concatenate((image_y, image_y_r, image_y_c), axis=0)
    image_z_combined = np.concatenate((image_z, image_z_r, image_z_c), axis=0)

    return image_x_combined, image_y_combined, image_z_combined


def func_file_save(save_dict, file_name, feature_package):
    image_x, image_y, image_z = feature_package
    Functions.image_save(image_x, os.path.join(save_dict, file_name + '_x.png'), dpi=300)
    Functions.image_save(image_y, os.path.join(save_dict, file_name + '_y.png'), dpi=300)
    Functions.image_save(image_z, os.path.join(save_dict, file_name + '_z.png'), dpi=300)


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, file_name + '_z.png')
    if os.path.exists(path_saved):
        return True
    return False


def process_all(fold=(0, 1)):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0] % 2)
    scan_name_list = get_all_scan_name()[fold[0]:: fold[1]]  # this name will not contain suffix
    save_image_dict = '/data_disk/CTA-CT_paired-dataset/' \
                      'registration_from_non_contrast_to_cta/visualization_optimal/semantics_and_ct_registered_3_view'

    for scan_name in Functions.iteration_with_time_bar(scan_name_list):
        if func_check_processed(save_image_dict, scan_name):
            continue
        else:
            image_x_combined, image_y_combined, image_z_combined = process_scan(scan_name)
            func_file_save(save_image_dict, scan_name, (image_x_combined, image_y_combined, image_z_combined))


if __name__ == '__main__':
    process_all((0, 2))
    exit()
    process_scan('Z154')
    exit()
