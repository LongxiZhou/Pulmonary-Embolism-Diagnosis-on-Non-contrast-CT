import Tool_Functions.Functions as Functions
import pe_dataset_management.basic_functions as basic_functions
import numpy as np
import os
import pe_dataset_management.inference_general_registration as apply_register
import Tool_Functions.performance_metrics as metrics
import format_convert.spatial_normalize as spatial_normalize
import format_convert.basic_transformations as basic_transformations
from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array \
    import show_jacobi_of_flow


def show_all_performance():

    dict_smooth_then_register = '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/' \
                                'smooth_flow_then_twice_register/performance'

    dict_direct_register = '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/optimal/performance'

    fn_list = os.listdir(dict_smooth_then_register)

    for fn in fn_list:
        path_twice = os.path.join(dict_smooth_then_register, fn)
        path_direct = os.path.join(dict_direct_register, fn)
        performance_twice = Functions.pickle_load_object(path_twice)
        performance_direct = Functions.pickle_load_object(path_direct)

        print('\n', fn)
        print(performance_direct)
        print(performance_twice)


def show_detailed_one_case(patient_id='patient-id-23716830'):

    """
    there may be very very little difference in dice score calculated in this function and in inference, because
    1) vessel mask to get registration flow is undergone bounding box clip (determined by lung) during inference, but
    here we did not use lung bounding box to clip
    2) there is an extra up-sample down-sample operation for registered mask in shape 256 in this function
    3) the mask for performance smooth is newly segmented after smooth registration in inference,
    while here we use the original mask
    """

    dict_cta, dict_non = basic_functions.find_patient_id_dataset_correspondence(patient_id, strip=True)
    print(dict_cta, dict_non)

    flow_direct = Functions.pickle_load_object('/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
                                               'optimal/registration_flow/' + patient_id + '.pickle')
    print("\njacobi analysis for flow direct:")
    show_jacobi_of_flow(flow_direct[1])

    performance_direct = Functions.pickle_load_object(
        '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
        'optimal/performance/' + patient_id + '.pickle')

    flow_smooth = Functions.pickle_load_object('/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
                                               'smooth_flow_then_twice_register/registration_flow/'
                                               + patient_id + '.pickle')

    print("\njacobi analysis for flow smooth stage two:")
    show_jacobi_of_flow(flow_smooth[1])
    print("\njacobi analysis for flow smooth overall:")
    show_jacobi_of_flow(flow_direct[1] + flow_smooth[1])

    performance_smooth = Functions.pickle_load_object(
        '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
        'smooth_flow_then_twice_register/performance/' + patient_id + '.pickle')

    print("\nperformance direct", performance_direct)
    print("performance smooth", performance_smooth)

    # rescaled_ct_fix (non-contrast)
    fixed_source = np.load(os.path.join(dict_non, 'rescaled_ct-denoise', patient_id + '.npz'))['array']
    # rescaled_ct_moving (CTA)
    if performance_direct['registration_conditions']['simulated_non_contrast']:
        moving_source = np.load(
            os.path.join(dict_cta, 'simulated_non_contrast/rescaled_ct-denoise', patient_id + '.npz'))['array']
    else:
        moving_source = np.load(os.path.join(dict_cta, 'rescaled_ct-denoise', patient_id + '.npz'))['array']

    mask_fix = np.load(os.path.join(dict_non, performance_direct['registration_conditions']['guide mask directory'],
                                    patient_id + '.npz'))['array']
    lung_fix = np.load(os.path.join(dict_non, 'semantics/lung_mask', patient_id + '.npz'))['array']
    if performance_direct['registration_conditions']['simulated_non_contrast']:
        mask_moving = np.load(os.path.join(
            dict_cta, 'simulated_non_contrast', performance_direct['registration_conditions']['guide mask directory'],
            patient_id + '.npz'))['array']
        lung_moving = np.load(os.path.join(dict_cta, 'simulated_non_contrast',
                                           'semantics/lung_mask', patient_id + '.npz'))['array']
    else:
        mask_moving = np.load(os.path.join(
            dict_cta, performance_direct['registration_conditions']['guide mask directory'],
            patient_id + '.npz'))['array']
        lung_moving = np.load(os.path.join(dict_cta,
                                           'semantics/lung_mask', patient_id + '.npz'))['array']

    mask_fix = apply_register.smooth_guide_mask(fixed_source, mask_fix)
    lung_fix = apply_register.smooth_guide_mask(None, lung_fix)
    mask_moving = apply_register.smooth_guide_mask(moving_source, mask_moving)
    lung_moving = apply_register.smooth_guide_mask(None, lung_moving)

    mask_fix_inside_lung = mask_fix * lung_fix
    print("ratio inside lung fix:", np.sum(mask_fix_inside_lung) / np.sum(mask_fix))
    mask_moving_inside_lung = mask_moving * lung_moving
    print("ratio inside lung moving:", np.sum(mask_moving_inside_lung) / np.sum(mask_moving))

    mask_fix_in_256 = spatial_normalize.rescale_to_new_shape(mask_fix, (256, 256, 256))
    mask_moving_in_256 = spatial_normalize.rescale_to_new_shape(mask_moving, (256, 256, 256))

    mask_fix_inside_lung_256 = spatial_normalize.rescale_to_new_shape(mask_fix_inside_lung, (256, 256, 256))
    mask_moving_inside_lung_256 = spatial_normalize.rescale_to_new_shape(mask_moving_inside_lung, (256, 256, 256))

    print("original 512", metrics.dice_score_two_class(mask_fix, mask_moving, simple=True))
    print("original 512 inside lung",
          metrics.dice_score_two_class(mask_fix_inside_lung, mask_moving_inside_lung, simple=True))
    print("original 256", metrics.dice_score_two_class(mask_fix_in_256, mask_moving_in_256, simple=True))
    print("original 256 inside lung",
          metrics.dice_score_two_class(mask_fix_inside_lung_256, mask_moving_inside_lung_256, simple=True))

    fixed_mask_normalized = basic_transformations.transformation_on_array(mask_fix, flow_direct[2])
    moving_mask_normalized = basic_transformations.transformation_on_array(mask_moving, flow_direct[0])

    fixed_mask_normalized_inside_lung = basic_transformations.transformation_on_array(
        mask_fix_inside_lung, flow_direct[2])
    moving_mask_normalized_inside_lung = basic_transformations.transformation_on_array(
        mask_moving_inside_lung, flow_direct[0])
    print("original normalized", metrics.dice_score_two_class(
        fixed_mask_normalized, moving_mask_normalized, simple=True))
    print("original normalized_inside_lung", metrics.dice_score_two_class(
        fixed_mask_normalized_inside_lung, moving_mask_normalized_inside_lung, simple=True))

    mask_direct_register = apply_register.register_with_flow_combine(mask_moving, flow_direct)
    mask_smooth_register = apply_register.register_with_flow_combine(mask_direct_register, flow_smooth)
    mask_direct_register_in_256 = spatial_normalize.rescale_to_new_shape(mask_direct_register, (256, 256, 256))
    mask_smooth_register_in_256 = spatial_normalize.rescale_to_new_shape(mask_smooth_register, (256, 256, 256))

    moving_source_direct_register = apply_register.register_with_flow_combine(moving_source, flow_direct)
    moving_source_smooth_register = apply_register.register_with_flow_combine(
        moving_source_direct_register, flow_smooth)

    print("direct 512", metrics.dice_score_two_class(mask_fix, mask_direct_register, simple=True))
    print("direct 256", metrics.dice_score_two_class(mask_fix_in_256, mask_direct_register_in_256, simple=True))
    print("smooth 512", metrics.dice_score_two_class(mask_fix, mask_smooth_register, simple=True))
    print("smooth 256", metrics.dice_score_two_class(mask_fix_in_256, mask_smooth_register_in_256, simple=True))

    mask_direct_register_inside_lung = apply_register.register_with_flow_combine(mask_moving_inside_lung, flow_direct)
    mask_direct_register_inside_lung_256 = spatial_normalize.rescale_to_new_shape(
        mask_direct_register_inside_lung, (256, 256, 256))
    mask_smooth_register_inside_lung = apply_register.register_with_flow_combine(
        mask_direct_register_inside_lung, flow_smooth)
    mask_smooth_register_inside_lung_256 = spatial_normalize.rescale_to_new_shape(
        mask_smooth_register_inside_lung, (256, 256, 256))

    print("direct 512 inside lung", metrics.dice_score_two_class(
        mask_fix_inside_lung, mask_direct_register_inside_lung, simple=True))
    print("direct 256 inside lung", metrics.dice_score_two_class(
        mask_fix_inside_lung_256, mask_direct_register_inside_lung_256, simple=True))
    print("smooth 512 inside lung", metrics.dice_score_two_class(
        mask_fix_inside_lung, mask_smooth_register_inside_lung, simple=True))
    print("smooth 256 inside lung", metrics.dice_score_two_class(
        mask_fix_inside_lung_256, mask_smooth_register_inside_lung_256, simple=True))

    mass_center = Functions.get_mass_center_for_binary(mask_fix, cast_to_int=True)

    # slice from z
    image_up = Functions.merge_image_with_mask(
        np.clip(fixed_source[:, :, mass_center[2]], -0.25, 0.7), mask_fix[:, :, mass_center[2]], show=False)
    image_mid = Functions.merge_image_with_mask(
        np.clip(moving_source_direct_register[:, :, mass_center[2]], -0.25, 0.7),
        mask_direct_register[:, :, mass_center[2]], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(moving_source_smooth_register[:, :, mass_center[2]], -0.25, 0.7),
        mask_smooth_register[:, :, mass_center[2]], show=False)

    image_z = np.concatenate([image_up, image_mid, image_down], axis=0)

    # slice from y
    image_up = Functions.merge_image_with_mask(
        np.clip(fixed_source[:, mass_center[1]], -0.25, 0.7), mask_fix[:, mass_center[1]], show=False)
    image_mid = Functions.merge_image_with_mask(
        np.clip(moving_source_direct_register[:, mass_center[1]], -0.25, 0.7),
        mask_direct_register[:, mass_center[1]], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(moving_source_smooth_register[:, mass_center[1]], -0.25, 0.7),
        mask_smooth_register[:, mass_center[1]], show=False)

    image_y = np.concatenate([image_up, image_mid, image_down], axis=0)

    # slice from x
    image_up = Functions.merge_image_with_mask(
        np.clip(fixed_source[mass_center[0]], -0.25, 0.7), mask_fix[mass_center[0]], show=False)
    image_mid = Functions.merge_image_with_mask(
        np.clip(moving_source_direct_register[mass_center[0]], -0.25, 0.7),
        mask_direct_register[mass_center[0]], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(moving_source_smooth_register[mass_center[0]], -0.25, 0.7),
        mask_smooth_register[mass_center[0]], show=False)

    image_x = np.concatenate([image_up, image_mid, image_down], axis=0)

    Functions.show_multiple_images(slice_x=image_x, slice_y=image_y, slice_z=image_z)


if __name__ == '__main__':
    # show_all_performance()
    show_detailed_one_case('patient-id-135')
    exit()
