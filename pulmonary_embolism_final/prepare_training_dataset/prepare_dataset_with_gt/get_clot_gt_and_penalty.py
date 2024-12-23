from analysis.get_surface_rim_adjacent_mean import get_surface
import pe_dataset_management.basic_functions as basic_functions
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import pe_dataset_management.registration.register_cta_to_ct.inference_fast_cta_to_ct as register_cta_to_non
import numpy as np
import os


def derive_penalty_array_and_clot_gt_non_from_clot_segmentation_on_cta(scan_name, extra_layer=2):
    """

    1) get clot mask on the paired CTA,
    2) register clot mask to non-contrast
    3) smooth the clot, remove clot outside vessel
    4) get false positive penalty array (as the registration introduce error, we give lower fp penalty around the clot)
    5) get clot annotation for non-contrast

    Note, false negative penalty array is all one

    :param extra_layer:
    :param scan_name
    :return: false_positive_penalty_array, clot_gt_probability_non in [512, 512, 512], numpy float32
    """
    clot_predict_cta, clot_gt_cta = get_clot_segmentation_cta(scan_name)

    # we add predict clot into gt, as the CTPA clot segmentation is trained on this, thus can provide smooth

    clot_predict_register_to_non, clot_gt_register_to_non = register_cta_to_non.cast_cta_to_non_contrast(
        [clot_predict_cta, clot_gt_cta], scan_name)

    clot_predict_register_to_non = smooth_mask(clot_predict_register_to_non)
    clot_gt_register_to_non = smooth_mask(clot_gt_register_to_non)

    clot_predict_non_extended = add_extra_layer(
        clot_predict_register_to_non, extra_layer=extra_layer, decay_for_extra_layer=0.5, strict_for_extra_layer=True)
    clot_gt_non_extended = add_extra_layer(
        clot_gt_register_to_non, extra_layer=extra_layer, decay_for_extra_layer=0.5, strict_for_extra_layer=True)

    if clot_gt_non_extended is not None:
        possible_clot_region = clot_gt_non_extended
    else:
        blood_vessel_high_recall_non = get_blood_vessel_high_recall_non(scan_name)
        clot_predict_non_extended = clot_predict_non_extended * blood_vessel_high_recall_non
        possible_clot_region = clot_predict_non_extended

    fp_penalty_array = 1 - np.array(possible_clot_region > 0, 'float32')

    if clot_gt_non_extended is not None:
        clot_gt_probability_non = clot_gt_non_extended
    else:
        clot_gt_probability_non = clot_predict_non_extended

    return fp_penalty_array, clot_gt_probability_non


def smooth_mask(binary_mask, cast_to_binary=True):
    if binary_mask is None:
        return None
    if cast_to_binary:
        binary_mask = np.array(binary_mask > 0.5, 'float32')
    binary_mask = binary_mask + get_surface(binary_mask, outer=True, strict=False)
    binary_mask = binary_mask - get_surface(binary_mask, outer=False, strict=False)
    return binary_mask


def add_extra_layer(binary_mask, extra_layer=2, decay_for_extra_layer=0.5,
                    strict_for_extra_layer=True):
    """

    extend the probability mask after registration

    :param binary_mask:
    :param extra_layer:
    :param decay_for_extra_layer:
    :param strict_for_extra_layer:
    :return: array in float32, range from 0 to 1
    """
    if binary_mask is None:
        return None

    temp_mask = np.array(binary_mask, 'float32')
    return_array = np.array(binary_mask, 'float32')

    for i in range(extra_layer):
        surface = get_surface(temp_mask, outer=True, strict=strict_for_extra_layer)
        temp_mask = temp_mask + surface
        return_array = return_array + surface * (decay_for_extra_layer ** (i + 1))

    return return_array


def get_blood_vessel_high_recall_non(scan_name):
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'
    data_dict_cta, data_dict_non_contrast = \
        basic_functions.find_patient_id_dataset_correspondence(scan_name=scan_name, strip=True)
    path_blood_high_recall = os.path.join(data_dict_non_contrast, 'semantics', 'blood_mask_high_recall', scan_name)

    return np.load(path_blood_high_recall)['array']


def get_clot_segmentation_cta(scan_name, show=False):
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    data_dict_cta, data_dict_non_contrast = \
        basic_functions.find_patient_id_dataset_correspondence(scan_name=scan_name, strip=True)

    manual_annotation_path = os.path.join(data_dict_cta, 'clot_gt', scan_name)
    prediction_path = os.path.join(data_dict_cta, 'semantics', 'blood_clot', scan_name)

    # get clot prediction
    predicted_clot_mask = np.load(prediction_path)['array']
    predicted_clot_mask = smooth_mask(predicted_clot_mask)

    # get artery mask
    artery_mask_direct_seg_path = os.path.join(data_dict_cta, 'semantics', 'artery_mask', scan_name)
    artery_mask_seg_on_simulated_non_path = os.path.join(
        data_dict_cta, 'simulated_non_contrast', 'semantics', 'artery_mask', scan_name)
    artery_mask_direct_seg = np.load(artery_mask_direct_seg_path)['array']
    artery_mask_seg_on_simulated_non = np.load(artery_mask_seg_on_simulated_non_path)['array']
    artery_mask = np.clip(artery_mask_direct_seg + artery_mask_seg_on_simulated_non, 0, 1)
    artery_mask = smooth_mask(artery_mask)

    clot_predict_cta = predicted_clot_mask * artery_mask
    clot_predict_cta = smooth_mask(clot_predict_cta)

    if os.path.exists(manual_annotation_path):
        manual_clot_mask = np.load(manual_annotation_path)['array']
        clot_gt_cta = manual_clot_mask
        clot_gt_cta = smooth_mask(clot_gt_cta)
        if show:
            print("dice, precision, recall between annotated clot and predicted clot (include vein fp):",
                  metrics.dice_score_two_class(predicted_clot_mask, clot_gt_cta))
            print("dice, precision, recall between annotated clot and predicted clot (exclude vein fp):",
                  metrics.dice_score_two_class(clot_predict_cta, clot_gt_cta))
    else:
        clot_gt_cta = None

    if show:
        rescaled_cta = np.load(os.path.join(data_dict_cta, 'rescaled_ct', scan_name))['array']
        if clot_gt_cta is not None:
            print("annotated clot")
            Functions.merge_image_with_mask_3d(rescaled_cta, clot_gt_cta, clip_window=(0.2, 0.8))
        print("predicted clot")
        Functions.merge_image_with_mask_3d(rescaled_cta, clot_predict_cta, clip_window=(0.2, 0.8))

    return clot_predict_cta, clot_gt_cta


def show_example_clot_gt_on_non_contrast(scan_name='patient-id-135.npz', extra_layer=0):
    # patient_id='Z102.npz', z = 227 is a very good example
    Functions.set_visible_device('1')

    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    dataset_cta, dataset_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    clot_predict = np.load(os.path.join(dataset_cta, 'semantics', 'blood_clot', scan_name))['array']
    if np.sum(clot_predict) == 0:
        print("no clot")
        exit()

    rescaled_ct = np.load(os.path.join(dataset_non, 'rescaled_ct', scan_name))['array']
    rescaled_cta = np.load(os.path.join(dataset_cta, 'rescaled_ct-denoise', scan_name))['array']

    rescaled_cta = register_cta_to_non.cast_cta_to_non_contrast(rescaled_cta, scan_name)

    fp_penalty_array, clot_gt_probability_non = \
        derive_penalty_array_and_clot_gt_non_from_clot_segmentation_on_cta(scan_name, extra_layer=extra_layer)

    loc_list = list(set(np.where(clot_gt_probability_non > 0.5)[2]))
    loc_list.sort()

    for z in loc_list[::3]:
        print(z)
        up = Functions.merge_image_with_mask_3d(
            rescaled_ct, clot_gt_probability_non, z_to_plot=z, clip_window=(-0.5, 0.7), show=False)
        down = Functions.merge_image_with_mask_3d(
            rescaled_cta, clot_gt_probability_non, z_to_plot=z, clip_window=(-0.5, 0.7), show=False)
        image = np.concatenate((up, down), axis=0)
        Functions.image_show(image)


if __name__ == '__main__':
    from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.stratify_gt_quality import \
        get_quality_of_scan_name

    fn_good_pair_good_registration, fn_good_pair_excellent_registration, \
        fn_excellent_pair_good_registration, fn_excellent_pair_excellent_registration = get_quality_of_scan_name()

    target_list = list(fn_excellent_pair_excellent_registration)
    target_list.sort()
    show_example_clot_gt_on_non_contrast(target_list[0], extra_layer=2)
