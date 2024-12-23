import sys

sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import os
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import reconstruct_semantic_from_sample_sequence
import numpy as np
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.performance.get_av_classification_metrics import analysis_clot_in_av
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
import pe_dataset_management.basic_functions as basic_functions
import Tool_Functions.performance_metrics as metrics
from analysis.other_functions import smooth_mask


def determine_model_path(patient_id=None, augment=False, use_real_clot=True):

    if patient_id is None:
        assert not use_real_clot
    if not use_real_clot:
        if not augment:
            return '/data_disk/pulmonary_embolism_final/check_point_dir/' \
                   'high_resolution/warm_up_simulation_only/vi_0.014_dice_0.720_precision_phase_model_guided.pth'
        else:
            return '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution_with_augment/' \
                   'warm_up_simulation_only/vi_0.014_dice_0.754_precision_phase_model_guided.pth'

    patient_id = Functions.strip_suffix(patient_id)
    ord_sum = 0
    for char in patient_id + '.pickle':
        ord_sum += ord(char)

    test_id = ord_sum % 5

    if not augment:
        top_dict_check_points = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution/'
    else:
        top_dict_check_points = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution_with_augment/'
    model_path = os.path.join(
        top_dict_check_points, 'with_annotation_test_id_' + str(test_id), 'best_model_guided.pth')

    return model_path


def segment_clot_prob_with_sample_sequence(
        sample_sequence, model=None, model_path=None, trim_length=4000, artery_mask=None):
    if model is None:
        if model_path is None:
            model_path = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution/' \
                         'with_annotation_test_id_0/vi_0.014_dice_0.568_precision_phase_model_guided.pth'
    if trim_length is None:
        trim_length = np.inf
    sample_sequence_with_clot = predict.predict_clot_for_sample_sequence(
        sample_sequence, model=model, model_path=model_path, min_depth=0.5, trim=True, trim_length=trim_length)

    clot_prob_predicted = reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_clot, (4, 4, 5), key='clot_prob_mask', background=0)

    if artery_mask is not None:
        clot_prob_predicted = clot_prob_predicted * artery_mask

    return clot_prob_predicted


def form_clot_gt_and_predicted_mask(patient_id='patient-id-135', visible_device='0', model=None, trim_length=4000,
                                    augment=False, use_real_clot=True):
    if patient_id[-7:] == '.pickle':
        patient_id = Functions.strip_suffix(patient_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism_final/' \
                                'training_samples_with_annotation/high_resolution/pe_ready_not_denoise'

    sample_path = os.path.join(top_dict_sample_sequence, patient_id + '.pickle')
    sample = Functions.pickle_load_object(sample_path)

    print("sample with key:", list(sample.keys()))
    for key, value in sample.items():
        if key == 'sample_sequence':
            print("sample sequence cube with key", list(sample[key][0].keys()))
            continue
        if key == 'center_line_loc_array':
            continue
        print("key:", key, "  value:", value)

    sample_sequence = sample['sample_sequence']
    model_path = determine_model_path(patient_id + '.pickle', augment=augment, use_real_clot=use_real_clot)
    print(model_path)

    dataset_dir_cta, dataset_dir_non = basic_functions.find_patient_id_dataset_correspondence(patient_id, strip=True)

    artery_mask = np.load(os.path.join(dataset_dir_non, 'semantics/artery_mask', patient_id + '.npz'))['array']
    vein_mask = np.load(os.path.join(dataset_dir_non, 'semantics/vein_mask', patient_id + '.npz'))['array']

    predict_clot_prob = segment_clot_prob_with_sample_sequence(sample_sequence, model, model_path,
                                                               trim_length=trim_length,
                                                               artery_mask=None)

    predict_clot_mask = np.array(predict_clot_prob > 0.5, 'float32') * artery_mask

    clot_gt_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence, (4, 4, 5), key='clot_gt_mask', background=0)
    clot_gt_mask = np.array(clot_gt_mask > 0.1, 'float32')

    clot_gt_mask = smooth_mask(clot_gt_mask, surface_add=0) * artery_mask

    gt_volume = np.sum(clot_gt_mask) * 334 / 512 * 334 / 512
    predicted_clot_binary_volume = np.sum(predict_clot_mask) * 334 / 512 * 334 / 512

    return_dict = {"gt_volume (mm^3)": gt_volume,
                   "predicted_clot_mask_volume (mm^3)": predicted_clot_binary_volume,
                   "region discovery (dice, recall, precision)": metrics.region_discovery_dice_3d(
                       predict_clot_mask, clot_gt_mask),
                   "(dice, recall, precision)": metrics.dice_score_two_class(
                       predict_clot_mask, clot_gt_mask)}

    a = analysis_clot_in_av(predict_clot_prob, artery_mask, vein_mask, None)

    return_dict["a-v clot ratio, artery clot average probability, artery clot probability volume in mm^3"] = a

    return_dict["registration_quality"] = sample["registration_quality"]
    return_dict["pe_pair_quality"] = sample["pe_pair_quality"]

    return return_dict


def predict_dataset(visible_device='0', trim_length=4000, augment=False, use_real_clot=True):
    tp_total = 0
    fp_total = 0
    fn_total = 0

    tp_total_rd = 0
    fp_total_rd = 0
    fn_total_rd = 0

    dice_list = []
    dice_rd_list = []
    recall_list = []
    recall_rd_list = []
    precision_list = []
    precision_rd_list = []

    fn_list = os.listdir('/data_disk/pulmonary_embolism_final/training_samples_with_annotation/'
                         'high_resolution/pe_ready_not_denoise')

    save_top_dict = '/data_disk/pulmonary_embolism_final/segmentation_performance/case_study_' + str(trim_length)
    if augment:
        save_top_dict = save_top_dict + '_augment'
    else:
        save_top_dict = save_top_dict + '_not_augment'
    if not use_real_clot:
        save_top_dict = save_top_dict + '_simulate_only'

    processed = 0

    for fn in fn_list:

        if augment:
            if not os.path.exists(determine_model_path(fn, augment, use_real_clot)):
                print("model not trained")
                processed += 1
                continue

        save_path = os.path.join(save_top_dict, fn)
        print("processing", fn, processed, '/', len(fn_list))
        if os.path.exists(save_path):
            value_dict = Functions.pickle_load_object(save_path)
        else:
            value_dict = form_clot_gt_and_predicted_mask(fn, visible_device, trim_length=trim_length, augment=augment,
                                                         use_real_clot=use_real_clot)

            Functions.pickle_save_object(save_path, value_dict)

        print(value_dict)

        dice, recall, precision = value_dict["(dice, recall, precision)"]
        gt_volume = value_dict['gt_volume (mm^3)']
        predict_volume = value_dict['predicted_clot_mask_volume (mm^3)']

        tp_total += recall * gt_volume
        fp_total += predict_volume - tp_total
        fn_total += gt_volume - tp_total
        dice_list.append(dice)
        precision_list.append(precision)
        recall_list.append(recall)

        dice_rd, recall_rd, precision_rd = value_dict['region discovery (dice, recall, precision)']
        tp_total_rd += recall_rd * gt_volume
        fp_total_rd += predict_volume - tp_total_rd
        fn_total_rd += gt_volume - tp_total_rd
        dice_rd_list.append(dice_rd)
        precision_rd_list.append(precision_rd)
        recall_rd_list.append(recall_rd)

        processed += 1

    overall_dice = tp_total * 2 / (tp_total * 2 + fp_total + fn_total)
    overall_dice_region_discovery = tp_total_rd * 2 / (tp_total_rd * 2 + fp_total_rd + fn_total_rd)

    return_dict = {"overall_dice": overall_dice,
                   "overall_dice_region_discovery": overall_dice_region_discovery,
                   "dice_list_each_case": dice_list,
                   "recall_list_each_case": recall_list,
                   "precision_list_each_case": precision_list,
                   "region_discovery_dice_list_each_case": dice_rd_list,
                   "region_discovery_recall_list_each_case": recall_rd_list,
                   "region_discovery_precision_list_each_case": precision_rd_list}

    save_path_overall_performance = '/data_disk/pulmonary_embolism_final/segmentation_performance/' + \
                                    "overall_performance_trim_" + str(trim_length)
    if augment:
        save_path_overall_performance = save_path_overall_performance + '_augment'
    else:
        save_path_overall_performance = save_path_overall_performance + '_not_augment'
    if not use_real_clot:
        save_path_overall_performance = save_path_overall_performance + '_simulate_only'

    save_path_overall_performance = save_path_overall_performance + '.pickle'

    Functions.pickle_save_object(save_path_overall_performance, return_dict)

    print(return_dict)


if __name__ == "__main__":
    predict_dataset(trim_length=4000, augment=True, use_real_clot=True)
    exit()
    predict_dataset(trim_length=4000, augment=False, use_real_clot=True)
    predict_dataset(trim_length=4000, augment=False, use_real_clot=False)
    predict_dataset(trim_length=4000, augment=True, use_real_clot=False)
    predict_dataset(trim_length=3000, augment=True, use_real_clot=True)
    predict_dataset(trim_length=3000, augment=False, use_real_clot=True)
    predict_dataset(trim_length=3000, augment=False, use_real_clot=False)
    predict_dataset(trim_length=3000, augment=True, use_real_clot=False)
    exit()
    import visualization.visualize_distribution.distribution_analysis as distribution
    value_dict = Functions.pickle_load_object(
        '/data_disk/pulmonary_embolism_final/segmentation_performance/overall_performance_trim_4000_not_augment.pickle')

    new_value_dict = {'Dice Each Patient': value_dict['dice_list_each_case'],
                      'Region Discovery Dice Each Patient': value_dict['region_discovery_dice_list_each_case']}

    distribution.distribution_plot(new_value_dict, nan_policy=None,
                                   save_path='/data_disk/pulmonary_embolism_final/pictures/not_augment_seg_dice.svg',
                                   title='Clot Segmentation Dice on Non-Contrast CT')
    exit()
    predict_dataset(trim_length=4000, augment=False, use_real_clot=False)
    predict_dataset(trim_length=4000, augment=True, use_real_clot=False)
