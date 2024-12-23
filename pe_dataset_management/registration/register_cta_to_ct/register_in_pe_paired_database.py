# based on the dataset: /data_disk/CTA-CT_paired-dataset
# input patient-id, condition, then will return corresponding files for generating registration field
# the registration field will map CTA to non-contrast

import os
import numpy as np
from pe_dataset_management.basic_functions import find_patient_id_dataset_correspondence, get_all_scan_name
import Tool_Functions.Functions as Functions
import format_convert.basic_transformations as basic_transform
from pe_dataset_management.inference_general_registration import normalization, register_with_given_flow, \
    register
from analysis.center_line_and_depth_3D import get_surface_distance, get_surface


def smooth_guide_mask(rescaled_ct, guide_mask):
    if rescaled_ct is not None:
        guide_mask = guide_mask * np.array(rescaled_ct > Functions.change_to_rescaled(-200), 'float32')
        # smooth
    guide_mask = guide_mask + get_surface.get_surface(guide_mask, outer=True, strict=False)
    guide_mask = guide_mask - get_surface.get_surface(guide_mask, outer=False, strict=False)

    return guide_mask


def get_data_for_registration(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset', simulated_non_contrast=True,
                              normalize=True, depth_guide=True,
                              guide_mask='semantics/blood_mask_high_recall',
                              mask_for_bounding_box='semantics/lung_mask', extend_guide_mask=0, smooth_mask=False,
                              other_condition=None):
    """

    :param other_condition: a dict, like {"apply_translate": True, "twice_register": False}
    :param smooth_mask:
    :param extend_guide_mask: add surface to the guide mask
    :param scan_name:
    :param top_dict:
    :param simulated_non_contrast:
    :param normalize:
    :param depth_guide: will be add to moving source and fixed source, like 'depth_and_center-line/depth_array'
    :param guide_mask:
    :param mask_for_bounding_box: like 'semantics/lung_mask'
    :return: all are numpy array in shape [256, 256, 256]
    moving_source, fixed_source, guide_mask_moving, guide_mask_fixed, flow_moving, flow_fix

    """

    # CTA --- moving,
    # non-contrast --- fixed

    if scan_name is not None:
        if len(scan_name) <= 4:
            scan_name = scan_name + '.npz'
        if len(scan_name) > 4:
            if not scan_name[-4:] == '.npz':
                scan_name = scan_name + '.npz'
    dataset_dict_cta, dataset_dict_non_contrast = \
        find_patient_id_dataset_correspondence(top_dict=top_dict, scan_name=scan_name, check_pair=False)[scan_name[:-4]]

    if other_condition is not None:
        if "apply_translate" in list(other_condition.keys()):
            if other_condition["apply_translate"] is False:
                if normalize is True:  # normalize here means apply_translate
                    print("set normalize from True to False")
                    normalize = False
        if "smooth_register" in list(other_condition.keys()):
            if other_condition["smooth_register"] is True:
                simulated_non_contrast = False
                # smooth register already applied the optimal registration to optimal condition
                dataset_dict_cta = os.path.join(dataset_dict_cta, "smooth_register")

    # rescaled_ct_fix (non-contrast)
    fixed_source = np.load(os.path.join(dataset_dict_non_contrast, 'rescaled_ct-denoise', scan_name))['array']
    # rescaled_ct_moving (CTA)
    if simulated_non_contrast:
        moving_source = np.load(
            os.path.join(dataset_dict_cta, 'simulated_non_contrast/rescaled_ct-denoise', scan_name))['array']
    else:
        moving_source = np.load(os.path.join(dataset_dict_cta, 'rescaled_ct-denoise', scan_name))['array']

    if guide_mask is not None:
        if simulated_non_contrast:
            guide_mask_moving_o = np.load(
                os.path.join(dataset_dict_cta, 'simulated_non_contrast', guide_mask, scan_name))['array']
        else:
            guide_mask_moving_o = np.load(os.path.join(dataset_dict_cta, guide_mask, scan_name))['array']
        guide_mask_fixed_o = np.load(os.path.join(dataset_dict_non_contrast, guide_mask, scan_name))['array']

        if smooth_mask:
            guide_mask_moving_o = smooth_guide_mask(moving_source, guide_mask_moving_o)
            guide_mask_fixed_o = smooth_guide_mask(fixed_source, guide_mask_fixed_o)

    else:
        guide_mask_moving_o = None
        guide_mask_fixed_o = None
        assert normalize is False

    for i in range(extend_guide_mask):
        if guide_mask_moving_o is not None:
            guide_mask_moving_o = guide_mask_moving_o + get_surface.get_surface(
                guide_mask_moving_o, outer=True, strict=False)

        if guide_mask_fixed_o is not None:
            guide_mask_fixed_o = guide_mask_fixed_o + get_surface.get_surface(
                guide_mask_fixed_o, outer=True, strict=False)

    if depth_guide is not None:
        if type(depth_guide) is str:
            if simulated_non_contrast:
                depth_moving = np.load(
                    os.path.join(dataset_dict_cta, 'simulated_non_contrast', depth_guide, scan_name))['array']
            else:
                depth_moving = np.load(os.path.join(dataset_dict_cta, depth_guide, scan_name))['array']

            depth_fixed = np.load(os.path.join(dataset_dict_non_contrast, depth_guide, scan_name))['array']

            depth_moving = depth_moving / np.max(depth_moving) * 2
            depth_fixed = depth_fixed / np.max(depth_fixed) * 2

            moving_source = moving_source + depth_moving
            fixed_source = fixed_source + depth_fixed
        else:
            assert type(depth_guide) is bool
            if depth_guide:
                # get depth_mask from guide_mask
                if guide_mask_moving_o is not None:
                    depth_moving, max_moving = get_surface_distance(
                        guide_mask_moving_o, strict=True, return_max_distance=True)
                    moving_source = moving_source + depth_moving / max_moving * 2
                if guide_mask_fixed_o is not None:
                    depth_fixed, max_fixed = get_surface_distance(
                        guide_mask_fixed_o, strict=True, return_max_distance=True)
                    fixed_source = fixed_source + depth_fixed / max_fixed * 2

    if mask_for_bounding_box is not None:
        if simulated_non_contrast:
            bound_moving = np.load(
                os.path.join(dataset_dict_cta, 'simulated_non_contrast', mask_for_bounding_box, scan_name))['array']
        else:
            bound_moving = np.load(os.path.join(dataset_dict_cta, mask_for_bounding_box, scan_name))['array']

        z_min, z_max = Functions.get_bounding_box(bound_moving, pad=3)[2]
        moving_source[:, :, 0: z_min] = 0
        moving_source[:, :, z_max:] = 0

        bound_fixed = np.load(os.path.join(dataset_dict_non_contrast, mask_for_bounding_box, scan_name))['array']
        z_min, z_max = Functions.get_bounding_box(bound_fixed, pad=3)[2]
        fixed_source[:, :, 0: z_min] = 0
        fixed_source[:, :, z_max:] = 0

    if normalize:
        guide_mask_fixed, flow_fix = normalization(guide_mask_fixed_o)
        guide_mask_moving, flow_moving = normalization(guide_mask_moving_o)
    else:
        flow_fix = [{"reshape": ((512, 512, 512), (256, 256, 256))}]
        flow_moving = [{"reshape": ((512, 512, 512), (256, 256, 256))}]
        guide_mask_moving = basic_transform.transformation_on_array(guide_mask_moving_o, flow_moving)
        guide_mask_fixed = basic_transform.transformation_on_array(guide_mask_fixed_o, flow_moving)

    moving_source = basic_transform.transformation_on_array(moving_source, flow_moving)
    fixed_source = basic_transform.transformation_on_array(fixed_source, flow_fix)

    return moving_source, fixed_source, guide_mask_moving, guide_mask_fixed, flow_moving, flow_fix


def get_register_flow_cta_to_non_contrast(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset',
                                          simulated_non_contrast=True,
                                          normalize=True, depth_guide=True,
                                          guide_mask='semantics/blood_mask_high_recall',
                                          mask_for_bounding_box=None, extend_guide_mask=0,
                                          use_guide_in_deep_learning=True, smooth_mask=False, other_condition=None,
                                          show_detail=False):
    """

    :param show_detail:
    :param other_condition: a dict, like {"apply_translate": True, "twice_register": False}
    :param smooth_mask:
    :param use_guide_in_deep_learning: whether input guide_mask into deep learning model
    :param extend_guide_mask: add surface to the guide mask
    :param scan_name:
    :param top_dict:
    :param simulated_non_contrast:
    :param normalize:
    :param depth_guide:
    :param guide_mask:
    :param mask_for_bounding_box:
    :return: [normalization_flow_cta, registration_flow, normalization_flow_non_contrast]
    """
    moving_source, fixed_source, guide_mask_moving, guide_mask_fixed, flow_moving, flow_fix = \
        get_data_for_registration(scan_name, top_dict=top_dict, simulated_non_contrast=simulated_non_contrast,
                                  normalize=normalize, depth_guide=depth_guide,
                                  guide_mask=guide_mask, mask_for_bounding_box=mask_for_bounding_box,
                                  extend_guide_mask=extend_guide_mask, smooth_mask=smooth_mask,
                                  other_condition=other_condition)

    if not use_guide_in_deep_learning:
        registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow = \
            register(moving_source, fixed_source, None, None, two_stage=False,
                     down_sample=False, return_flow=True)
    else:
        registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow = \
            register(moving_source, fixed_source, guide_mask_moving, guide_mask_fixed, two_stage=False,
                     down_sample=False, return_flow=True)

    if not use_guide_in_deep_learning:
        registered_seg = register_with_given_flow(guide_mask_moving, register_flow)

    if other_condition is not None and "twice_register" in list(other_condition.keys()):
        if other_condition["twice_register"] is True:
            import pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array as\
                smooth_operations
            smoothed_flow = smooth_operations.blur_flow_with_convolution_kernel(
                register_flow, blur_kernel_radius=20, blur_parameter=2, blur_type='half_decay')

            if show_detail:
                print("\ndice for registration first stage:", compute_dice(registered_seg, fixed_seg), '\n')
                print("jacobi for register flow first stage")
                smooth_operations.show_jacobi_of_flow(register_flow)
                print("jacobi for register flow first stage smoothed")
                smooth_operations.show_jacobi_of_flow(smoothed_flow)

            moving_source_registered_once = register_with_given_flow(moving_source, smoothed_flow)
            guide_mask_moving_registered_once = register_with_given_flow(guide_mask_moving, smoothed_flow)

            if not use_guide_in_deep_learning:
                registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow_2 = \
                    register(moving_source_registered_once, fixed_source, None, None, two_stage=False,
                             down_sample=False, return_flow=True)
            else:
                registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow_2 = \
                    register(moving_source_registered_once, fixed_source, guide_mask_moving_registered_once,
                             guide_mask_fixed, two_stage=False,
                             down_sample=False, return_flow=True)

            register_flow = smoothed_flow + register_flow_2

            if show_detail:
                print("\ndice for registration first stage smooth:",
                      compute_dice(guide_mask_moving_registered_once, fixed_seg), '\n')
                print("\ndice for registration second stage:", compute_dice(registered_seg, fixed_seg), '\n')
                print("jacobi for register flow second stage")
                smooth_operations.show_jacobi_of_flow(register_flow_2)

                print("jacobi for register flow final register flow")
                smooth_operations.show_jacobi_of_flow(register_flow)

                print("show loss in dice caused by interpolation during two stage registration",
                      1 - compute_dice(registered_seg, register_with_given_flow(guide_mask_moving, register_flow)))

            if not use_guide_in_deep_learning:
                registered_seg = register_with_given_flow(guide_mask_moving_registered_once, register_flow)

    registration_conditions = {"simulated_non_contrast": simulated_non_contrast,
                               "normalize": normalize, "depth_guide": depth_guide,
                               "guide mask directory": guide_mask, "mask_for_bounding_box": mask_for_bounding_box,
                               "extend_guide_mask": extend_guide_mask, "smooth_mask": smooth_mask}

    if other_condition is not None:
        registration_conditions["other_condition"] = other_condition

    return [flow_moving, register_flow, flow_fix], \
        {"guide mask dice on 256 original": compute_dice(guide_mask_moving, guide_mask_fixed),
            "guide mask dice on 256 registered": compute_dice(guide_mask_fixed, registered_seg),
            "registration_conditions": registration_conditions}


def register_cta_to_non_contrast_with_flow_combine(array_in_cta, registration_flow_combine):
    """

    :param registration_flow_combine: [normalization_flow_cta, registration_flow, normalization_flow_non_contrast]
    :param array_in_cta: numpy array in shape (512, 512, 512)
    :return: array_registered_to_non_contrast, numpy array in shape (512, 512, 512)
    """

    if array_in_cta is None:
        return None

    normalization_flow_cta, registration_flow, normalization_flow_non_contrast = registration_flow_combine

    normalized_array = basic_transform.transformation_on_array(array_in_cta, normalization_flow_cta, reverse=False)

    # registered to normalized non-contrast, normalized array in shape (256, 256, 256)
    # flow in shape (1, 3, 256, 256, 256)
    registered_array = register_with_given_flow(normalized_array, registration_flow)

    # undo normalization
    array_registered_to_non_contrast = basic_transform.transformation_on_array(
        registered_array, normalization_flow_non_contrast, reverse=True)

    return array_registered_to_non_contrast


def compute_dice(pre, mask):
    return 2 * np.sum(pre * mask) / (np.sum(pre * pre) + np.sum(mask * mask))


def get_save_path_flow_and_performance(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset/'
                                                           'registration_from_cta_to_non_contrast', version='v1'):
    if len(scan_name) > 4:
        if scan_name[-4:] == '.npz':
            scan_name = scan_name[:-4]
    save_path_flow = os.path.join(top_dict, version, 'registration_flow', scan_name + '.pickle')
    save_path_performance = os.path.join(top_dict, version, 'performance', scan_name + '.pickle')

    return save_path_flow, save_path_performance


def get_registration_pickle_v1(scan_name,
                               top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                               other_condition=None):

    if other_condition is not None:
        if other_condition["twice_register"] and not other_condition["smooth_register"]:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'direct_twice_register')
        else:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'smooth_flow_then_twice_register')
    else:
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'v1')

    if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
        print(scan_name, 'v1 processed, performance:')
        performance_dict = Functions.pickle_load_object(save_path_performance)
        print(performance_dict)
        return None

    # do not change these settings
    flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
        scan_name=scan_name, simulated_non_contrast=False, mask_for_bounding_box=None,
        guide_mask='semantics/blood_mask', extend_guide_mask=0, use_guide_in_deep_learning=True,
        smooth_mask=True, depth_guide=True, other_condition=other_condition)

    print("v1 performance for", scan_name)
    print(performance_dict)

    Functions.pickle_save_object(save_path_flow, flow_combined)
    Functions.pickle_save_object(save_path_performance, performance_dict)


def get_registration_pickle_v2(scan_name,
                               top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                               other_condition=None):
    # compared to v1, use simulated_non_contrast

    if other_condition is not None:
        if other_condition["twice_register"] and not other_condition["smooth_register"]:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'direct_twice_register')
        else:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'smooth_flow_then_twice_register')
    else:
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'v2')

    if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
        print(scan_name, 'v2 processed, performance:')
        performance_dict = Functions.pickle_load_object(save_path_performance)
        print(performance_dict)
        return None

    # do not change these settings
    flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
        scan_name=scan_name, simulated_non_contrast=True, mask_for_bounding_box=None,
        guide_mask='semantics/blood_mask', extend_guide_mask=0, use_guide_in_deep_learning=True,
        smooth_mask=True, depth_guide=True, other_condition=other_condition)

    print("v2 performance for", scan_name)
    print(performance_dict)

    Functions.pickle_save_object(save_path_flow, flow_combined)
    Functions.pickle_save_object(save_path_performance, performance_dict)


def get_registration_pickle_v3(scan_name,
                               top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                               other_condition=None):
    # compared to v1, use 'semantics/lung_mask' as bounding box

    if other_condition is not None:
        if other_condition["twice_register"] and not other_condition["smooth_register"]:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'direct_twice_register')
        else:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'smooth_flow_then_twice_register')
    else:
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'v3')

    if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
        print(scan_name, 'v3 processed, performance:')
        performance_dict = Functions.pickle_load_object(save_path_performance)
        print(performance_dict)
        return None

    # do not change these settings
    flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
        scan_name=scan_name, simulated_non_contrast=False, mask_for_bounding_box='semantics/lung_mask',
        guide_mask='semantics/blood_mask', extend_guide_mask=0, use_guide_in_deep_learning=True,
        smooth_mask=True, depth_guide=True, other_condition=other_condition)

    print("v3 performance for", scan_name)
    print(performance_dict)

    Functions.pickle_save_object(save_path_flow, flow_combined)
    Functions.pickle_save_object(save_path_performance, performance_dict)


def get_registration_pickle_v4(scan_name,
                               top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                               other_condition=None):
    # compared to v1, use 'semantics/blood_mask_high_recall' as guide_mask

    if other_condition is not None:
        if other_condition["twice_register"] and not other_condition["smooth_register"]:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'direct_twice_register')
        else:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'smooth_flow_then_twice_register')
    else:
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'v4')

    if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
        print(scan_name, 'v4 processed, performance:')
        performance_dict = Functions.pickle_load_object(save_path_performance)
        print(performance_dict)
        return None

    # do not change these settings
    flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
        scan_name=scan_name, simulated_non_contrast=False, mask_for_bounding_box=None,
        guide_mask='semantics/blood_mask_high_recall', extend_guide_mask=0, use_guide_in_deep_learning=True,
        smooth_mask=True, depth_guide=True, other_condition=other_condition)

    print("v4 performance for", scan_name)
    print(performance_dict)

    Functions.pickle_save_object(save_path_flow, flow_combined)
    Functions.pickle_save_object(save_path_performance, performance_dict)


def get_registration_pickle_v5(scan_name,
                               top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                               other_condition=None):
    # compared to v1, use 'semantics/blood_mask_high_recall' as guide_mask and use simulated_non_contrast

    if other_condition is not None:
        if other_condition["twice_register"] and not other_condition["smooth_register"]:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'direct_twice_register')
        else:
            save_path_flow, save_path_performance = get_save_path_flow_and_performance(
                scan_name, top_dict, 'smooth_flow_then_twice_register')
    else:
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'v5')

    if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
        print(scan_name, 'v5 processed, performance:')
        performance_dict = Functions.pickle_load_object(save_path_performance)
        print(performance_dict)
        return None

    # do not change these settings
    flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
        scan_name=scan_name, simulated_non_contrast=True, mask_for_bounding_box=None,
        guide_mask='semantics/blood_mask_high_recall', extend_guide_mask=0, use_guide_in_deep_learning=True,
        smooth_mask=True, depth_guide=True, other_condition=other_condition)

    print("v5 performance for", scan_name)
    print(performance_dict)

    Functions.pickle_save_object(save_path_flow, flow_combined)
    Functions.pickle_save_object(save_path_performance, performance_dict)


def processing_registration_database(fold=(0, 4),
                                     top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast'):
    scan_name_list = get_all_scan_name()
    scan_name_list = Functions.split_list_by_ord_sum(scan_name_list, fold)
    processed_count = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed_count, '/', len(scan_name_list))
        save_path_flow, save_path_performance = get_save_path_flow_and_performance(scan_name, top_dict, 'optimal')
        if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
            print(scan_name, 'optimal existed, performance:')
            performance_dict = Functions.pickle_load_object(save_path_performance)
            print(performance_dict)
            processed_count += 1
            continue
        get_registration_pickle_v1(scan_name, top_dict=top_dict)
        get_registration_pickle_v2(scan_name, top_dict=top_dict)
        get_registration_pickle_v3(scan_name, top_dict=top_dict)
        get_registration_pickle_v4(scan_name, top_dict=top_dict)
        get_registration_pickle_v5(scan_name, top_dict=top_dict)
        processed_count += 1


# register twice, but the mask will not re-segment. we find this is not good.
def processing_second_stage_registration_database(
        fold=(0, 4), top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast'):
    scan_name_list = get_all_scan_name()
    scan_name_list = Functions.split_list_by_ord_sum(scan_name_list, fold)

    def load_optimal_performance_condition(scan):
        save_path_performance = os.path.join(top_dict, "optimal", "performance", scan + '.pickle')
        current_performance_dict = Functions.pickle_load_object(save_path_performance)
        return current_performance_dict

    other_condition = {"apply_translate": True, "twice_register": True}

    processed_count = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed_count, '/', len(scan_name_list))
        current_optimal_condition = load_optimal_performance_condition(scan_name)
        print("current optimal condition is:", current_optimal_condition["optimal_version"])
        print("current optimal performance:", current_optimal_condition)
        if current_optimal_condition["optimal_version"] == 'v1':
            get_registration_pickle_v1(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v2':
            get_registration_pickle_v2(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v3':
            get_registration_pickle_v3(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v4':
            get_registration_pickle_v4(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v5':
            get_registration_pickle_v5(scan_name, top_dict=top_dict, other_condition=other_condition)
        else:
            raise ValueError("wrong optimal condition", current_optimal_condition)

        processed_count += 1


# register twice, and the mask will be re-segment. but only for the second stage
def processing_smooth_registration_database(
        fold=(0, 4), top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
        process_needed=True):

    # TODO this function is not complete!

    if process_needed:
        import pe_dataset_management.ct_cta_paired_dataset.establish_smooth_registered_cta as pre_register
        pre_register.process_all(fold)
        import pe_dataset_management.ct_cta_paired_dataset.establish_semantics as get_semantics
        get_semantics.prepare_semantic_for_twice_register(
            fold, top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA')
    scan_name_list = get_all_scan_name()
    scan_name_list = Functions.split_list_by_ord_sum(scan_name_list, fold)

    def load_optimal_performance_condition(scan):
        save_path_performance = os.path.join(top_dict, "optimal", "performance", scan + '.pickle')
        current_performance_dict = Functions.pickle_load_object(save_path_performance)
        return current_performance_dict

    other_condition = {"apply_translate": False, "twice_register": False, "smooth_register": True}

    processed_count = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed_count, '/', len(scan_name_list))
        current_optimal_condition = load_optimal_performance_condition(scan_name)
        print("current optimal condition is:", current_optimal_condition["optimal_version"])
        print("current optimal performance:", current_optimal_condition)
        if current_optimal_condition["optimal_version"] == 'v1':
            get_registration_pickle_v1(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v2':
            get_registration_pickle_v2(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v3':
            get_registration_pickle_v3(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v4':
            get_registration_pickle_v4(scan_name, top_dict=top_dict, other_condition=other_condition)
        elif current_optimal_condition["optimal_version"] == 'v5':
            get_registration_pickle_v5(scan_name, top_dict=top_dict, other_condition=other_condition)
        else:
            raise ValueError("wrong optimal condition", current_optimal_condition)

        processed_count += 1


def establish_optimal_registration(top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast',
                                   fold=(0, 1)):

    def load_performance_for_version(scan, version='v1'):
        save_path_performance = os.path.join(top_dict, version, "performance", scan + '.pickle')
        current_performance_dict = Functions.pickle_load_object(save_path_performance)
        return current_performance_dict

    def load_flow_for_version(scan, version='v1'):
        save_path_flow = os.path.join(top_dict, version, "registration_flow", scan + '.pickle')
        registration_flow = Functions.pickle_load_object(save_path_flow)
        return registration_flow

    def clean_flow_for_version(scan):
        for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
            save_path_flow = os.path.join(top_dict, version, "registration_flow", scan + '.pickle')
            if os.path.exists(save_path_flow):
                print("removing:", save_path_flow)
                os.remove(save_path_flow)

    def check_optimal_exists(scan):
        save_path_flow = os.path.join(top_dict, "optimal", "registration_flow", scan + '.pickle')
        save_path_performance = os.path.join(top_dict, "optimal", "performance", scan + '.pickle')
        if os.path.exists(save_path_flow) and os.path.exists(save_path_performance):
            print(scan, 'optimal existed, performance:')
            performance = Functions.pickle_load_object(save_path_performance)
            print(performance)
            return True
        return False

    def save_optimal_registration(scan, current_performance_dict, current_optimal_version):
        assert current_performance_dict is not None
        assert current_optimal_version is not None
        save_path_flow = os.path.join(top_dict, "optimal", "registration_flow", scan + '.pickle')
        save_path_performance = os.path.join(top_dict, "optimal", "performance", scan + '.pickle')

        if os.path.exists(save_path_performance):
            previous_optimal = Functions.pickle_load_object(save_path_performance)
            if current_performance_dict['guide mask dice on 256 registered'] <= \
                    previous_optimal['guide mask dice on 256 registered']:
                print(scan, 'contains previously non-inferior registration for version',
                      current_optimal_version, 'performance dict:\n',
                      previous_optimal)
                if "optimal_version" not in list(previous_optimal.keys()):
                    Functions.pickle_save_object(save_path_performance, current_performance_dict)
                return None
            else:
                print("\nsaving optimal registration for", scan, "optimal version is:", current_optimal_version)
                print(current_performance_dict)
                Functions.pickle_save_object(save_path_performance, current_performance_dict)
                optimal_flow = load_flow_for_version(scan, current_optimal_version)
                Functions.pickle_save_object(save_path_flow, optimal_flow)
        else:
            print("\nsaving optimal registration for", scan, "optimal version is:", current_optimal_version)
            print(current_performance_dict)
            Functions.pickle_save_object(save_path_performance, current_performance_dict)
            optimal_flow = load_flow_for_version(scan, current_optimal_version)
            Functions.pickle_save_object(save_path_flow, optimal_flow)

    scan_name_list = get_all_scan_name()
    scan_name_list = Functions.split_list_by_ord_sum(scan_name_list, fold)
    processed_count = 0

    optimal_version = None
    optimal_performance = 0
    optimal_performance_dict = None
    version_list = ['v1', 'v2', 'v3', 'v4', 'v5']

    for scan_name in scan_name_list:
        print("processing:", scan_name, processed_count, '/', len(scan_name_list))
        if check_optimal_exists(scan_name):
            clean_flow_for_version(scan_name)
            processed_count += 1
            continue
        # get optimal version
        for version_register in version_list:
            performance_dict = load_performance_for_version(scan_name, version_register)
            if performance_dict['guide mask dice on 256 registered'] > optimal_performance:
                optimal_version = version_register
                optimal_performance = performance_dict['guide mask dice on 256 registered']
                optimal_performance_dict = performance_dict
                optimal_performance_dict["optimal_version"] = version_register

        save_optimal_registration(scan_name, optimal_performance_dict, optimal_version)
        optimal_version = None
        optimal_performance = 0
        optimal_performance_dict = None
        clean_flow_for_version(scan_name)
        processed_count += 1


def load_performance_cta_to_non_contrast(scan_name, dict_flow='/data_disk/CTA-CT_paired-dataset/'
                                                              'registration_from_cta_to_non_contrast/optimal'):
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    performance_path = os.path.join(dict_flow, 'performance', scan_name[:-4] + '.pickle')
    performance_dict = Functions.pickle_load_object(performance_path)

    return performance_dict


def load_flow_cta_to_non_contrast(scan_name, dict_flow='/data_disk/CTA-CT_paired-dataset/'
                                                       'registration_from_cta_to_non_contrast/optimal',
                                  performance_dict_only=False):
    if performance_dict_only:
        return load_performance_cta_to_non_contrast(scan_name, dict_flow=dict_flow)

    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    flow_path = os.path.join(dict_flow, 'registration_flow', scan_name[:-4] + '.pickle')
    flow_combined = Functions.pickle_load_object(flow_path)

    performance_path = os.path.join(dict_flow, 'performance', scan_name[:-4] + '.pickle')
    performance_dict = Functions.pickle_load_object(performance_path)

    return flow_combined, performance_dict


def reproduce_optimal_registration_on_vessel(scan_name='Z138.npz'):

    import pe_dataset_management.basic_functions as basic_functions

    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    sub_dataset_cta, sub_dataset_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    flow_combined, performance_dict = load_flow_cta_to_non_contrast(scan_name)

    optimal_condition = performance_dict['registration_conditions']

    vessel_path_cta = sub_dataset_cta
    if optimal_condition['simulated_non_contrast']:
        vessel_path_cta = os.path.join(vessel_path_cta, 'simulated_non_contrast')
    vessel_path_cta = os.path.join(vessel_path_cta, optimal_condition['guide mask directory'], scan_name)

    vessel_path_non = os.path.join(sub_dataset_non, optimal_condition['guide mask directory'], scan_name)

    blood_mask_cta = np.load(vessel_path_cta)['array']
    blood_mask_non_contrast = np.load(vessel_path_non)['array']

    print("dice vessel 512 original", compute_dice(blood_mask_cta, blood_mask_non_contrast))

    # flow_combined, performance_dict = get_register_flow_cta_to_non_contrast(
    #     scan_name=scan_name, simulated_non_contrast=False, mask_for_bounding_box=None,
    #     guide_mask='semantics/blood_mask', extend_guide_mask=0, use_guide_in_deep_learning=True,
    #     smooth_mask=True, depth_guide=True)

    print(performance_dict)

    blood_mask_cta_registered = register_cta_to_non_contrast_with_flow_combine(blood_mask_cta, flow_combined)
    # blood_mask_cta_registered = temp_func(blood_mask_cta, flow_combined, blood_mask_non_contrast)

    print("dice vessel 512 registered", compute_dice(blood_mask_cta_registered, blood_mask_non_contrast))

    rescaled_ct_non_contrast = np.load(sub_dataset_non +
                                       '/rescaled_ct-denoise/' + scan_name)['array']

    rescaled_ct_non_contrast = np.clip(rescaled_ct_non_contrast + 0.5, 0, 1.2)

    mid_z = int(np.median(np.where(blood_mask_non_contrast > 0.5)[2]))

    Functions.merge_image_with_mask(rescaled_ct_non_contrast[:, :, mid_z], blood_mask_cta_registered[:, :, mid_z])
    Functions.merge_image_with_mask(rescaled_ct_non_contrast[:, :, mid_z], blood_mask_non_contrast[:, :, mid_z])


def update_registration_database(fold=(0, 1),
                                 top_dict='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast'):
    processing_registration_database(fold, top_dict=top_dict)
    establish_optimal_registration(top_dict=top_dict, fold=fold)
    # processing_smooth_registration_database(fold, top_dict=top_dict, process_needed=True)


if __name__ == '__main__':

    Functions.set_visible_gpu('1')
    update_registration_database((0, 4))
