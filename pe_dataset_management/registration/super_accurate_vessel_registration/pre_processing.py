"""
Background:

even with perfect registration, the registered vessel and fixed vessel will not have dice = 1,
because there are segmentation error for vessels.

the registration will align lung mask, big vessels satisfactorily.

registration will make the segmentation for vessel more difficult.


Solution:
1) normalize the non and CTA, and get flow from CTA to non, and non to CTA.
2) smooth the flow, and divide the flows by half, then apply corresponding flow to CTA_simulated_non and apply to non.
This step, register CTA_simulated_non and non in the middle state, denote as CTA_hr, non_hr
3) lung mask and blood vessel outside lung is the union of CTA_hr and non_hr.
lung mask will be used to segment vessels
inside lung, vessel outside lung will be used to add landmark for registration (after segment vessel)
4) get vessel probability inside lung, with predict_rescaled.get_prediction_blood_vessel, for CTA_hr and non_hr
5) register CTA_hr to non_hr, based non vessel inside lung. get CTA_hrr
6) register CTA_hrr to non
"""
import os
import pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database as register_cta_to_non
import pe_dataset_management.registration.register_ct_to_cta.register_in_pe_paired_database as register_non_to_cta
import pe_dataset_management.basic_functions as basic_functions
from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array import \
    blur_flow_with_convolution_kernel
from pe_dataset_management.inference_general_registration import register_with_given_flow

import format_convert.basic_transformations as basic_transform
import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.center_line_and_depth_3D as get_depth
import numpy as np
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import format_convert.spatial_normalize as spatial_normalize


def load_data_and_get_hr(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset/', blur_parameter=2,
                         blur_kernel_radius=20, blur_type='half_decay', show_performance=True):
    assert type(scan_name) is str
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    # load flow
    dict_flow_non_to_cta = os.path.join(top_dict, 'registration_from_non_contrast_to_cta/optimal')
    dict_flow_cta_to_non = os.path.join(top_dict, 'registration_from_cta_to_non_contrast/optimal')

    flow_combined_non_to_cta, performance_dict_non_to_cta = register_non_to_cta.load_flow_non_contrast_to_cta(
        scan_name, dict_flow=dict_flow_non_to_cta)
    flow_combined_cta_to_non, performance_dict_cta_to_non = register_cta_to_non.load_flow_cta_to_non_contrast(
        scan_name, dict_flow=dict_flow_cta_to_non)
    normalize_flow_cta, _, normalize_flow_non = flow_combined_cta_to_non
    flow_cta_to_non = flow_combined_cta_to_non[1]
    flow_non_to_cta = flow_combined_non_to_cta[1]

    def modify_normalize_flow(normalize_flow):
        # normalize_flow like [{'reshape': ((512, 512, 512), (256, 256, 256))}, {'translate': (-2.0, -11.0, -11.0)}]
        # do not reshape, only translate
        translate_vector_on_256 = normalize_flow[1]['translate']
        translate_vector_on_512 = [2 * translate_vector_on_256[0],
                                   2 * translate_vector_on_256[1], 2 * translate_vector_on_256[2]]
        new_flow = [{'translate': translate_vector_on_512}]
        return new_flow

    # do not reshape from (512, 512, 512) to (256, 256, 256)
    normalize_flow_cta = modify_normalize_flow(normalize_flow_cta)
    normalize_flow_non = modify_normalize_flow(normalize_flow_non)

    # smooth flow
    if blur_parameter > 0:
        flow_cta_to_non = blur_flow_with_convolution_kernel(
            flow_cta_to_non, blur_kernel_radius=blur_kernel_radius, blur_parameter=blur_parameter, blur_type=blur_type)
        flow_non_to_cta = blur_flow_with_convolution_kernel(
            flow_non_to_cta, blur_kernel_radius=blur_kernel_radius, blur_parameter=blur_parameter, blur_type=blur_type)

    # half divide flow
    flow_cta_to_non_h = flow_cta_to_non / 2
    flow_non_to_cta_h = flow_non_to_cta / 2

    # reshape flow
    def up_sample_flow(flow_256):
        flow_512 = np.zeros((1, 3, 512, 512, 512), 'float32')
        for axis in range(3):
            flow_512[0, axis] = spatial_normalize.rescale_to_new_shape(
                flow_256[0, axis], (512, 512, 512), change_format=False)
        return flow_512

    flow_cta_to_non_h = up_sample_flow(flow_cta_to_non_h)
    flow_non_to_cta_h = up_sample_flow(flow_non_to_cta_h)

    # load masks and rescaled ct
    dataset_cta, dataset_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    lung_mask_cta = np.load(
        os.path.join(dataset_cta, 'simulated_non_contrast/semantics/lung_mask', scan_name))['array']
    vessel_cta = np.load(
        os.path.join(dataset_cta, 'simulated_non_contrast/semantics/blood_mask_high_recall', scan_name))['array']
    rescaled_cta = np.load(os.path.join(dataset_cta, 'simulated_non_contrast/rescaled_ct-denoise', scan_name))['array']

    lung_mask_non = np.load(os.path.join(dataset_non, 'semantics/lung_mask', scan_name))['array']
    vessel_non = np.load(
        os.path.join(dataset_non, 'semantics/blood_mask_high_recall', scan_name))['array']
    rescaled_non = np.load(os.path.join(dataset_non, 'rescaled_ct-denoise', scan_name))['array']

    # cast masks and rescaled ct to standard space
    vessel_cta_normalize = basic_transform.transformation_on_array(vessel_cta, normalize_flow_cta, reverse=False)
    vessel_non_normalize = basic_transform.transformation_on_array(vessel_non, normalize_flow_non, reverse=False)

    lung_cta_normalize = basic_transform.transformation_on_array(lung_mask_cta, normalize_flow_cta, reverse=False)
    lung_non_normalize = basic_transform.transformation_on_array(lung_mask_non, normalize_flow_non, reverse=False)

    cta_normalized = basic_transform.transformation_on_array(rescaled_cta, normalize_flow_cta, reverse=False)
    non_normalized = basic_transform.transformation_on_array(rescaled_non, normalize_flow_non, reverse=False)

    # apply flow
    vessel_cta_hr = register_with_given_flow(vessel_cta_normalize, flow_cta_to_non_h)
    lung_cta_hr = register_with_given_flow(lung_cta_normalize, flow_cta_to_non_h)
    cta_hr = register_with_given_flow(cta_normalized, flow_cta_to_non_h)

    vessel_non_hr = register_with_given_flow(vessel_non_normalize, flow_non_to_cta_h)
    lung_non_hr = register_with_given_flow(lung_non_normalize, flow_non_to_cta_h)
    non_hr = register_with_given_flow(non_normalized, flow_non_to_cta_h)

    def show_performance_hr():
        print("performance cta to non", performance_dict_cta_to_non)
        print("performance non to cta", performance_dict_non_to_cta)

        print("dice lung original:", metrics.dice_score_two_class(lung_non_normalize, lung_cta_normalize))
        print("dice lung half meet:", metrics.dice_score_two_class(lung_non_hr, lung_cta_hr))

        print("dice vessel original:", metrics.dice_score_two_class(vessel_non_normalize, vessel_cta_normalize))
        print("dice vessel half meet:", metrics.dice_score_two_class(vessel_non_hr, vessel_cta_hr))

        image_up = Functions.merge_image_with_mask(
            np.clip(non_hr[:, :, 256], -0.25, 0.7), vessel_non_hr[:, :, 256], show=False)
        image_down = Functions.merge_image_with_mask(
            np.clip(cta_hr[:, :, 256], -0.25, 0.7), vessel_cta_hr[:, :, 256], show=False)

        image = np.concatenate((image_up, image_down), axis=0)
        Functions.image_show(image)

        image_up = Functions.merge_image_with_mask(
            np.clip(non_hr[:, :, 256], -0.25, 0.7), lung_non_hr[:, :, 256], show=False)
        image_down = Functions.merge_image_with_mask(
            np.clip(cta_hr[:, :, 256], -0.25, 0.7), lung_cta_hr[:, :, 256], show=False)

        image = np.concatenate((image_up, image_down), axis=0)
        Functions.image_show(image)

    if show_performance:
        show_performance_hr()

    return non_hr, cta_hr, vessel_non_hr, vessel_cta_hr, lung_non_hr, lung_cta_hr, \
        normalize_flow_cta, normalize_flow_non


def prepare_data_for_next_stage(non_hr, cta_hr, vessel_non_hr, vessel_cta_hr, lung_non_hr, lung_cta_hr, visualize=True,
                                exclude_outside_lung=False):
    import basic_tissue_prediction.predict_rescaled as predict_rescaled
    import analysis.connectivity_refine_fast as connectivity_refine

    # process lung mask
    lung_cta_hr, weight_for_vessel_cta = smooth_lung_and_get_weight_for_voxel_nearly_lung_boundary(lung_cta_hr)
    lung_non_hr, weight_for_vessel_non = smooth_lung_and_get_weight_for_voxel_nearly_lung_boundary(lung_non_hr)
    lung_mask_intersect = np.array(lung_non_hr * lung_cta_hr > 0.5, 'float32')
    lung_mask_combined = np.clip(lung_non_hr + lung_cta_hr, 0, 1)
    depth_lung_mask_non = get_depth.get_surface_distance(lung_non_hr, strict=False)
    depth_lung_mask_cta = get_depth.get_surface_distance(lung_cta_hr, strict=False)

    if visualize:
        print("dice high recall vessel", metrics.dice_score_two_class(vessel_cta_hr, vessel_non_hr))
        print("dice high recall vessel in lung",
              metrics.dice_score_two_class(vessel_cta_hr * lung_mask_intersect, vessel_non_hr * lung_mask_intersect))

    # segment vessel
    vessel_seg_non_hr = predict_rescaled.get_prediction_blood_vessel(
        non_hr, lung_mask=lung_mask_combined, probability_only=True, batch_size=16)
    vessel_seg_non_hr = predict_rescaled.surrounding_mean_convolution(vessel_seg_non_hr)
    vessel_seg_cta_hr = predict_rescaled.get_prediction_blood_vessel(
        cta_hr, lung_mask=lung_mask_combined, probability_only=True, batch_size=16)
    vessel_seg_cta_hr = predict_rescaled.surrounding_mean_convolution(vessel_seg_cta_hr)
    
    # refine vessel step 1
    vessel_seg_non_hr = connectivity_refine.select_region(
        np.array(vessel_seg_non_hr > 0.25, 'float32'), leave_count=2, min_ratio=0.2)
    vessel_seg_non_hr = vessel_seg_non_hr + get_surface.get_surface(vessel_seg_non_hr, outer=True, strict=False)
    vessel_seg_non_hr = vessel_seg_non_hr - get_surface.get_surface(vessel_seg_non_hr, outer=False, strict=False)

    vessel_seg_cta_hr = connectivity_refine.select_region(
        np.array(vessel_seg_cta_hr > 0.25, 'float32'), leave_count=2, min_ratio=0.2)
    vessel_seg_cta_hr = vessel_seg_cta_hr + get_surface.get_surface(vessel_seg_cta_hr, outer=True, strict=False)
    vessel_seg_cta_hr = vessel_seg_cta_hr - get_surface.get_surface(vessel_seg_cta_hr, outer=False, strict=False)

    # refine vessel step 2
    protect_region_non = np.array(vessel_seg_non_hr > 0.5, 'float32')
    protect_region_cta = np.array(vessel_seg_cta_hr > 0.5, 'float32')
    for i in range(3):
        protect_region_cta = protect_region_cta - get_surface.get_surface(protect_region_cta, outer=False, strict=False)
        protect_region_non = protect_region_non - get_surface.get_surface(protect_region_non, outer=False, strict=False)
    for i in range(3):
        protect_region_cta = protect_region_cta + get_surface.get_surface(protect_region_cta, outer=True, strict=False)
        protect_region_non = protect_region_non + get_surface.get_surface(protect_region_non, outer=True, strict=False)
    valid_region_cta = np.clip(np.array(depth_lung_mask_cta > 5, 'float32') + protect_region_cta, 0, 1)
    valid_region_non = np.clip(np.array(depth_lung_mask_non > 5, 'float32') + protect_region_non, 0, 1)
    vessel_seg_cta_hr = valid_region_cta * vessel_seg_cta_hr
    vessel_seg_non_hr = valid_region_non * vessel_seg_non_hr

    if visualize:
        print("dice predicted vessel", metrics.dice_score_two_class(vessel_seg_non_hr, vessel_seg_cta_hr))
        print("dice in lung", metrics.dice_score_two_class(
            vessel_seg_non_hr * lung_mask_intersect, vessel_seg_cta_hr * lung_mask_intersect))
        print("volume in lung non", np.sum(lung_mask_intersect * vessel_seg_non_hr),
              "volume total non", np.sum(vessel_seg_non_hr))
        print("volume in lung cta", np.sum(lung_mask_intersect * vessel_seg_cta_hr),
              "volume total cta", np.sum(vessel_seg_cta_hr))

    if visualize:
        import visualization.visualize_3d.visualize_stl as stl
        stl.visualize_numpy_as_stl(vessel_seg_non_hr)
        stl.visualize_numpy_as_stl(vessel_seg_cta_hr)

    # get landmark
    depth_array_vessel_non, max_depth_non = get_depth.get_surface_distance(vessel_seg_non_hr, return_max_distance=True)
    depth_array_vessel_cta, max_depth_cta = get_depth.get_surface_distance(vessel_seg_cta_hr, return_max_distance=True)

    # add landmark
    non_hr = non_hr + depth_array_vessel_non / max_depth_non
    cta_hr = cta_hr + depth_array_vessel_cta / max_depth_cta

    if exclude_outside_lung:
        # get vessel inside lung (vessel outside lung will decay by 0.75^distance_to_lung)
        vessel_seg_non_hr = vessel_seg_non_hr * weight_for_vessel_non
        vessel_seg_cta_hr = vessel_seg_cta_hr * weight_for_vessel_cta

    if visualize:
        print("dice vessel in lung:", metrics.dice_score_two_class(
            vessel_seg_non_hr * lung_non_hr, vessel_seg_cta_hr * lung_cta_hr))

        # visualize lung mask union
        image_up = Functions.merge_image_with_mask(
            np.clip(non_hr[:, :, 256], -0.25, 0.7), lung_non_hr[:, :, 256], show=False)
        image_down = Functions.merge_image_with_mask(
            np.clip(cta_hr[:, :, 256], -0.25, 0.7), lung_cta_hr[:, :, 256], show=False)
        image = np.concatenate((image_up, image_down), axis=0)
        Functions.image_show(image)

        # visualize vessel seg
        image_up = Functions.merge_image_with_mask(
            np.clip(non_hr[:, :, 256], -0.25, 0.7), vessel_seg_non_hr[:, :, 256], show=False)
        image_down = Functions.merge_image_with_mask(
            np.clip(cta_hr[:, :, 256], -0.25, 0.7), vessel_seg_cta_hr[:, :, 256], show=False)
        image = np.concatenate((image_up, image_down), axis=0)
        Functions.image_show(image)

    return non_hr, cta_hr, vessel_seg_non_hr, vessel_seg_cta_hr, lung_non_hr, lung_cta_hr


def generate_data_for_scan(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset/', blur_parameter=2,
                           blur_kernel_radius=20, blur_type='half_decay', visualize=True):
    if visualize:
        print("Loading data and get half way registration")
    non_hr, cta_hr, vessel_non_hr, vessel_cta_hr, lung_non_hr, lung_cta_hr, \
        normalize_flow_cta, normalize_flow_non = load_data_and_get_hr(
            scan_name, top_dict, blur_parameter, blur_kernel_radius, blur_type, visualize)

    if visualize:
        print("Predicting vessels in lung")
    non_hr, cta_hr, vessel_seg_non_hr, vessel_seg_cta_hr, lung_non_hr, lung_cta_hr = prepare_data_for_next_stage(
        non_hr, cta_hr, vessel_non_hr, vessel_cta_hr, lung_non_hr, lung_cta_hr, visualize)

    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'non_hr.npy', non_hr, compress=False)
    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'cta_hr.npy', cta_hr, compress=False)
    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'vessel_seg_non_hr.npz', vessel_seg_non_hr, compress=True)
    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'vessel_seg_cta_hr.npz', vessel_seg_cta_hr, compress=True)
    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'lung_non_hr.npz', lung_non_hr, compress=True)
    Functions.save_np_array('/data_disk/pulmonary_registration/super_precision_register/sample_data',
                            'lung_cta_hr.npz', lung_cta_hr, compress=True)

    return non_hr, vessel_seg_non_hr, cta_hr, vessel_seg_cta_hr, lung_non_hr, lung_cta_hr


def smooth_lung_and_get_weight_for_voxel_nearly_lung_boundary(lung_mask, cast_to_binary=True):
    lung_mask = smooth_mask(lung_mask, cast_to_binary=cast_to_binary)
    weight_for_vessel = np.array(lung_mask)
    temp_array = np.array(lung_mask)

    for i in range(7):
        surface_temp = get_surface.get_surface(temp_array, outer=True, strict=False)
        temp_array = temp_array + surface_temp
        weight_for_vessel = weight_for_vessel + surface_temp * (0.75 ** (i + 1))

    return lung_mask, weight_for_vessel


def smooth_mask(binary_mask, cast_to_binary=True):
    if cast_to_binary:
        binary_mask = np.array(binary_mask > 0.5, 'float32')

    binary_mask = binary_mask + get_surface.get_surface(binary_mask, outer=True, strict=False)
    binary_mask = binary_mask + get_surface.get_surface(binary_mask, outer=True, strict=False)
    binary_mask = binary_mask - get_surface.get_surface(binary_mask, outer=False, strict=False)
    binary_mask = binary_mask - get_surface.get_surface(binary_mask, outer=False, strict=False)

    return binary_mask


if __name__ == '__main__':
    generate_data_for_scan(scan_name='Z211', visualize=True, blur_parameter=2)
