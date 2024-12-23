"""
here we only load the optimal flow then make inference

if the optimal flow is not calculated, update with
 .register_cta_to_ct.register_in_pe_paired_database.update_registration_database()
"""

import pe_dataset_management.basic_functions as basic_functions
import pe_dataset_management.registration.register_ct_to_cta.register_in_pe_paired_database as registration_operations
import numpy as np


def cast_non_contrast_to_cta(input_array_or_array_list, scan_name,
                             flow_top_dict='/data_disk/CTA-CT_paired-dataset/'
                                           'registration_from_non_contrast_to_cta/optimal', show_performance=False,
                             smooth=False, blur_parameter=1, blur_kernel_radius=10, blur_type='half_decay'):
    """

    :param blur_type:
    :param blur_kernel_radius:
    :param blur_parameter:
    :param smooth: smooth_the_flow
    :param show_performance:
    :param input_array_or_array_list: numpy array in shape [512, 512, 512] or array list
    :param scan_name:
    :param flow_top_dict:
    :return: registered array in shape [512, 512, 512]
    """
    if show_performance:
        print(load_performance_cta_to_non_contrast(scan_name, flow_top_dict))

    flow_combined, performance_dict = registration_operations.load_flow_non_contrast_to_cta(
        scan_name, dict_flow=flow_top_dict)

    if smooth:
        from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array import blur_flow_with_convolution_kernel
        flow_combined[1] = blur_flow_with_convolution_kernel(
            flow_combined[1], blur_kernel_radius=blur_kernel_radius,
            blur_parameter=blur_parameter, blur_type=blur_type)

    if not type(input_array_or_array_list) is list:
        input_array = input_array_or_array_list
        assert np.shape(input_array) == (512, 512, 512)
        registered = registration_operations.register_non_contrast_to_cta_with_flow_combine(input_array, flow_combined)
        return registered

    else:
        registered_list = []
        for input_array in input_array_or_array_list:
            registered = registration_operations.register_non_contrast_to_cta_with_flow_combine(input_array,
                                                                                                flow_combined)
            registered_list.append(registered)

        return registered_list


def load_performance_cta_to_non_contrast(scan_name, dict_flow='/data_disk/CTA-CT_paired-dataset/'
                                                              'registration_from_cta_to_non_contrast/optimal'):
    import os
    import Tool_Functions.Functions as Functions
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    performance_path = os.path.join(dict_flow, 'performance', scan_name[:-4] + '.pickle')
    performance_dict = Functions.pickle_load_object(performance_path)

    return performance_dict


def example_cast_non_to_cta(scan_id='patient-id-24296069', smooth=True):
    import Tool_Functions.Functions as Functions
    import Tool_Functions.performance_metrics as metrics
    import format_convert.basic_transformations as basic_transform
    import os

    dataset_cta, dataset_non = basic_functions.find_patient_id_dataset_correspondence(scan_id, strip=True)

    vessel_cta = np.load(
        os.path.join(dataset_cta, 'simulated_non_contrast/semantics/blood_mask_high_recall', scan_id + '.npz'))['array']
    rescaled_cta = np.load(os.path.join(dataset_cta, 'rescaled_ct-denoise', scan_id + '.npz'))['array']

    vessel_non = np.load(
        os.path.join(dataset_non, 'semantics/blood_mask_high_recall', scan_id + '.npz'))['array']
    rescaled_non = np.load(os.path.join(dataset_non, 'rescaled_ct-denoise', scan_id + '.npz'))['array']

    vessel_non_r, rescaled_non_r = cast_non_contrast_to_cta([vessel_non, rescaled_non], scan_id, show_performance=True,
                                                            smooth=smooth)

    flow_combined, performance_dict = registration_operations.load_flow_non_contrast_to_cta(
        scan_id, dict_flow='/data_disk/CTA-CT_paired-dataset/'
                           'registration_from_non_contrast_to_cta/v2')

    normalization_flow_non, _, normalization_flow_cta = flow_combined
    vessel_cta_normalize = basic_transform.transformation_on_array(vessel_cta, normalization_flow_cta, reverse=False)
    vessel_non_normalize = basic_transform.transformation_on_array(vessel_non, normalization_flow_non, reverse=False)

    print("original dice:", metrics.dice_score_two_class(vessel_non_normalize, vessel_cta_normalize))

    print("registration for vessel", metrics.dice_score_two_class(vessel_non_r, vessel_cta))
    image_up = Functions.merge_image_with_mask(
        np.clip(rescaled_cta[:, :, 256], -0.25, 0.7), vessel_cta[:, :, 256], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(rescaled_non_r[:, :, 256], -0.25, 0.7), vessel_non_r[:, :, 256], show=False)

    image = np.concatenate((image_up, image_down), axis=0)
    Functions.image_show(image)


if __name__ == '__main__':

    example_cast_non_to_cta(scan_id='Z211', smooth=True)

    exit()

    name_list = basic_functions.get_all_scan_name()
    for name in name_list:
        print(name)
        print(load_performance_cta_to_non_contrast(name))
