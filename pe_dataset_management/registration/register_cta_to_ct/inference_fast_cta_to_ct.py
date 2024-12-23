"""
here we only load the optimal flow then make inference

if the optimal flow is not calculated, update with
 .register_cta_to_ct.register_in_pe_paired_database.update_registration_database()
"""

import pe_dataset_management.basic_functions as basic_functions
import pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database as registration_operations
import numpy as np


def cast_cta_to_non_contrast(input_array_or_array_list, scan_name,
                             flow_top_dict='/data_disk/CTA-CT_paired-dataset/'
                                           'registration_from_cta_to_non_contrast/optimal', show_performance=False,
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
    :return: registered array or array list in shape [512, 512, 512]
    """
    if show_performance:
        print(load_performance_cta_to_non_contrast(scan_name, flow_top_dict))

    flow_combined, performance_dict = registration_operations.load_flow_cta_to_non_contrast(
        scan_name, dict_flow=flow_top_dict)

    if smooth:
        from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array import \
            blur_flow_with_convolution_kernel
        flow_combined[1] = blur_flow_with_convolution_kernel(
            flow_combined[1], blur_kernel_radius=blur_kernel_radius,
            blur_parameter=blur_parameter, blur_type=blur_type)

    if not type(input_array_or_array_list) is list:
        input_array = input_array_or_array_list
        assert np.shape(input_array) == (512, 512, 512)
        registered = registration_operations.register_cta_to_non_contrast_with_flow_combine(input_array, flow_combined)
        return registered

    else:
        registered_list = []
        for input_array in input_array_or_array_list:
            registered = registration_operations.register_cta_to_non_contrast_with_flow_combine(input_array,
                                                                                                flow_combined)
            registered_list.append(registered)

        return registered_list


def cast_cta_to_non_contrast_two_stage(
        input_array_or_array_list, scan_name,
        flow_top_dict_stage_one_optimal='/data_disk/CTA-CT_paired-dataset/'
                                        'registration_from_cta_to_non_contrast/optimal',
        flow_top_dict_stage_two='/data_disk/CTA-CT_paired-dataset/'
                                'registration_from_cta_to_non_contrast/smooth_flow_then_twice_register'):
    """

    :param input_array_or_array_list: numpy array in shape [512, 512, 512] or array list
    :param scan_name:
    :param flow_top_dict_stage_one_optimal:
    :param flow_top_dict_stage_two:
    :return: registered array or array list in shape [512, 512, 512]
    """
    flow_combined_stage_one, performance_dict_stage_one = registration_operations.load_flow_cta_to_non_contrast(
        scan_name, dict_flow=flow_top_dict_stage_one_optimal)

    flow_combined_stage_two, performance_dict_stage_two = registration_operations.load_flow_cta_to_non_contrast(
        scan_name, dict_flow=flow_top_dict_stage_two)

    if not type(input_array_or_array_list) is list:
        input_array = input_array_or_array_list
        assert np.shape(input_array) == (512, 512, 512)
        registered_stage_one = registration_operations.register_cta_to_non_contrast_with_flow_combine(
            input_array, flow_combined_stage_one)
        registered_stage_two = registration_operations.register_cta_to_non_contrast_with_flow_combine(
            registered_stage_one, flow_combined_stage_two)
        return registered_stage_two

    else:
        registered_list = []
        for input_array in input_array_or_array_list:
            assert np.shape(input_array) == (512, 512, 512)
            registered_stage_one = registration_operations.register_cta_to_non_contrast_with_flow_combine(
                input_array, flow_combined_stage_one)
            registered_stage_two = registration_operations.register_cta_to_non_contrast_with_flow_combine(
                registered_stage_one, flow_combined_stage_two)
            registered_list.append(registered_stage_two)

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


def example_cast_cta_to_non(scan_id='patient-id-135', smooth=True):
    import Tool_Functions.Functions as Functions
    import Tool_Functions.performance_metrics as metrics
    import format_convert.basic_transformations as basic_transform
    import pe_dataset_management.inference_general_registration as apply_register
    import os

    dataset_cta, dataset_non = basic_functions.find_patient_id_dataset_correspondence(scan_id, strip=True)

    vessel_cta = np.load(
        os.path.join(dataset_cta, 'simulated_non_contrast/semantics/blood_mask_high_recall', scan_id + '.npz'))['array']

    lung_mask_cta = np.load(
        os.path.join(dataset_cta, 'simulated_non_contrast/semantics/lung_mask', scan_id + '.npz'))['array']

    rescaled_cta = np.load(os.path.join(dataset_cta, 'rescaled_ct-denoise', scan_id + '.npz'))['array']

    vessel_non = np.load(
        os.path.join(dataset_non, 'semantics/blood_mask_high_recall', scan_id + '.npz'))['array']

    lung_mask_non = np.load(
        os.path.join(dataset_non, 'semantics/lung_mask', scan_id + '.npz'))['array']

    rescaled_non = np.load(os.path.join(dataset_non, 'rescaled_ct-denoise', scan_id + '.npz'))['array']

    vessel_cta = apply_register.smooth_guide_mask(rescaled_cta, vessel_cta)
    lung_mask_cta = apply_register.smooth_guide_mask(None, lung_mask_cta)
    vessel_non = apply_register.smooth_guide_mask(rescaled_non, vessel_non)
    lung_mask_non = apply_register.smooth_guide_mask(None, lung_mask_non)

    vessel_cta_r, rescaled_cta_r, lung_cta_r = cast_cta_to_non_contrast(
        [vessel_cta, rescaled_cta, lung_mask_cta], scan_id, show_performance=True,
        smooth=smooth, blur_parameter=3, blur_kernel_radius=10)

    vessel_cta_r_2, rescaled_cta_r_2, lung_cta_r_2 = cast_cta_to_non_contrast_two_stage(
        [vessel_cta, rescaled_cta, lung_mask_cta], scan_id)

    flow_combined, performance_dict = registration_operations.load_flow_cta_to_non_contrast(
        scan_id, dict_flow='/data_disk/CTA-CT_paired-dataset/'
                           'registration_from_cta_to_non_contrast/optimal')

    normalization_flow_cta, _, normalization_flow_non = flow_combined
    vessel_cta_normalize = basic_transform.transformation_on_array(vessel_cta, normalization_flow_cta, reverse=False)
    vessel_non_normalize = basic_transform.transformation_on_array(vessel_non, normalization_flow_non, reverse=False)

    lung_cta_normalize = basic_transform.transformation_on_array(lung_mask_cta, normalization_flow_cta, reverse=False)
    lung_non_normalize = basic_transform.transformation_on_array(lung_mask_non, normalization_flow_non, reverse=False)

    print("original dice vessel (256):",
          metrics.dice_score_two_class(vessel_non_normalize, vessel_cta_normalize, simple=True))

    print("original dice lung (256):",
          metrics.dice_score_two_class(lung_non_normalize, lung_cta_normalize, simple=True))

    print("registration for vessel (512)", metrics.dice_score_two_class(vessel_cta_r, vessel_non, simple=True))
    print("registration for vessel two stage (512)",
          metrics.dice_score_two_class(vessel_cta_r_2, vessel_non, simple=True))

    print("registration for lung (512)",
          metrics.dice_score_two_class(lung_cta_r, lung_mask_non, simple=True))
    print("registration for lung two stage (512)",
          metrics.dice_score_two_class(lung_cta_r_2, lung_mask_non, simple=True))

    image_up = Functions.merge_image_with_mask(
        np.clip(rescaled_non[:, :, 256], -0.25, 0.7), vessel_non[:, :, 256], show=False)
    image_mid = Functions.merge_image_with_mask(
        np.clip(rescaled_cta_r[:, :, 256], -0.25, 0.7), vessel_cta_r[:, :, 256], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(rescaled_cta_r_2[:, :, 256], -0.25, 0.7), vessel_cta_r_2[:, :, 256], show=False)

    image = np.concatenate((image_up, image_mid, image_down), axis=0)

    Functions.image_show(image)


if __name__ == '__main__':

    example_cast_cta_to_non(scan_id='Z211', smooth=True)

    exit()

    name_list = basic_functions.get_all_scan_name()
    for name in name_list:
        print(name)
        print(load_performance_cta_to_non_contrast(name))
