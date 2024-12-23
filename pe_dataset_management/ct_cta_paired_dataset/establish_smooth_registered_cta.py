"""
Smooth the optimal registration flow, and apply to CTA, which enables multiple registration
"""
import pe_dataset_management.basic_functions as basic_functions
import pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database as register_inference
import Tool_Functions.Functions as Functions
import os
import pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array as smooth_flow
import format_convert.basic_transformations as basic_transform
import numpy as np


def smooth_register_rescaled_cta(scan_name):
    """

    :param scan_name:
    :return:
    """
    if len(scan_name) >= 4:
        assert not scan_name[:-4] == '.npz'

    # check process
    data_dict_cta, data_dict_non_contrast = \
        basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)
    save_dict = os.path.join(data_dict_cta, 'smooth_register', 'rescaled_ct-denoise')
    if os.path.exists(os.path.join(save_dict, scan_name + '.npz')):
        print("processed")
        return None

    # get flow
    flow_combined_cta_to_non, performance_dict = register_inference.load_flow_cta_to_non_contrast(scan_name)
    print("performance for", scan_name, performance_dict)

    normalize_flow_cta, registration_flow_cta_to_non, normalize_flow_non = flow_combined_cta_to_non

    def modify_normalize_flow(normalize_flow):
        # normalize_flow like [{'reshape': ((512, 512, 512), (256, 256, 256))}, {'translate': (-2.0, -11.0, -11.0)}]
        # do not reshape, only translate
        translate_vector_on_256 = normalize_flow[1]['translate']
        translate_vector_on_512 = [2 * translate_vector_on_256[0],
                                   2 * translate_vector_on_256[1], 2 * translate_vector_on_256[2]]
        new_flow = [{'translate': translate_vector_on_512}]
        return new_flow

    # smooth flow and process flow
    normalize_flow_cta = modify_normalize_flow(normalize_flow_cta)
    normalize_flow_non = modify_normalize_flow(normalize_flow_non)

    registration_flow_cta_to_non = smooth_flow.smooth_256_then_up_sample_to_512(
        registration_flow_cta_to_non, blur_kernel_radius=20, blur_parameter=2,
        blur_type='half_decay', show_jacobi=False)

    # load rescaled CTA
    if performance_dict['registration_conditions']['simulated_non_contrast']:
        path_cta = os.path.join(data_dict_cta, 'simulated_non_contrast', 'rescaled_ct-denoise', scan_name + '.npz')
    else:
        path_cta = os.path.join(data_dict_cta, 'rescaled_ct-denoise', scan_name + '.npz')
    rescaled_cta_for_optimal_registration = np.load(path_cta)['array']

    # register CTA to non contrast
    cta_normalized = basic_transform.transformation_on_array(
        rescaled_cta_for_optimal_registration, normalize_flow_cta, reverse=False)
    cta_registered = register_inference.register_with_given_flow(cta_normalized, registration_flow_cta_to_non)
    cta_registered = basic_transform.transformation_on_array(
        cta_registered, normalize_flow_non, reverse=True)

    # save_registered array
    Functions.save_np_array(save_dict, scan_name + '.npz', cta_registered, compress=True, dtype='float16')
    visualize_effect(scan_name, save_image=True, show_image=False)
    return None


def process_all(fold=(0, 1)):
    scan_name_list = basic_functions.get_all_scan_name()[fold[0]:: fold[1]]
    processed = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed, '/', len(scan_name_list))
        smooth_register_rescaled_cta(scan_name)
        processed += 1


def visualize_effect(scan_name, save_image=False, show_image=True):
    data_dict_cta, data_dict_non_contrast = \
        basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    save_path_registered = os.path.join(data_dict_cta, 'smooth_register', 'rescaled_ct-denoise', scan_name + '.npz')
    cta_smooth_registered = np.load(save_path_registered)['array']

    original_cta_path = os.path.join(data_dict_cta, 'rescaled_ct-denoise', scan_name + '.npz')
    cta_not_registered = np.load(original_cta_path)['array']

    rescaled_non = np.load(os.path.join(data_dict_non_contrast, 'rescaled_ct-denoise', scan_name + '.npz'))['array']

    image = np.zeros([512, 1536], 'float32')
    image[:, 0: 512] = cta_smooth_registered[:, :, 256]
    image[:, 512: 1024] = rescaled_non[:, :, 256]
    image[:, 1024:] = cta_not_registered[:, :, 256]
    image = np.clip(image, Functions.change_to_rescaled(-1000), Functions.change_to_rescaled(600))

    if show_image:
        Functions.image_show(image, gray=True)

    if save_image:
        image_save_path = os.path.join(
            data_dict_cta, 'smooth_register', 'visualization', 'smooth_register_effect', scan_name + '.png')
        Functions.image_save(image, image_save_path, gray=True, dpi=300)


if __name__ == '__main__':
    current_fold = (0, 4)
    process_all(current_fold)
