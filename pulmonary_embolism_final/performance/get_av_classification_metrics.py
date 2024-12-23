import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
from chest_ct_database.public_datasets.RAD_ChestCT_dataset import load_func_for_ct
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import \
    reconstruct_semantic_from_sample_sequence, convert_ct_into_tubes
import analysis.center_line_and_depth_3D as get_depth
import os
import numpy as np
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
import pe_dataset_management.basic_functions as basic_functions


# must conform with ./prepare_training_dataset/./form_raw_dataset
absolute_cube_length = (4, 4, 5)
min_depth_get_sequence = 0.5
exclude_center_out = True


trim_length = 3000
augment = False


if augment:
    model_path_simulation_only = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution_with_augment/' \
                                 'warm_up_simulation_only/vi_0.015_dice_0.792_precision_phase_model_guided.pth'
    model_path_with_gt = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution_with_augment/' \
                         'with_annotation/stable_phase/vi_0.015_dice_0.635_precision_phase_model_guided.pth'
else:
    model_path_simulation_only = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution/' \
                                 'warm_up_simulation_only/vi_0.014_dice_0.720_precision_phase_model_guided.pth'
    model_path_with_gt = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution/' \
                         'with_annotation_test_id_0/vi_0.014_dice_0.568_precision_phase_model_guided.pth'


def get_dict(simulation_only=True):
    save_top_dict = '/data_disk/pulmonary_embolism_final/statistics'

    if augment:
        folder_metrics = 'augment_trim_' + str(trim_length)
    else:
        folder_metrics = 'not_augment_trim_' + str(trim_length)

    if simulation_only:
        save_top_dict = os.path.join(save_top_dict, 'simulation_only_' + folder_metrics)
        model_path = model_path_simulation_only
    else:
        save_top_dict = os.path.join(save_top_dict, 'with_gt_' + folder_metrics)
        model_path = model_path_with_gt

    return save_top_dict, model_path


def predict_all_rad(simulation_only=True, fold=(0, 1)):
    save_top_dict, model_path = get_dict(simulation_only)

    model = predict.load_saved_model_guided(model_path=model_path)

    scan_name_list = os.listdir('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise')[fold[0]:: fold[1]]
    processed = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed, '/', len(scan_name_list))
        get_prediction_clot_rad(scan_name, model=model, top_dict_save=save_top_dict)
        processed += 1


def predict_all_paired_dataset(simulation_only=True, fold=(0, 1)):
    save_top_dict, model_path = get_dict(simulation_only)

    model = predict.load_saved_model_guided(model_path=model_path)

    scan_name_list = basic_functions.get_all_scan_name()[fold[0]:: fold[1]]
    processed = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed, '/', len(scan_name_list))
        get_prediction_clot_paired_dataset(scan_name, model=model, top_dict_save=save_top_dict)
        processed += 1


def predict_all_chinese_dataset(simulation_only=True, fold=(0, 1), dataset='All'):
    save_top_dict, model_path = get_dict(simulation_only)

    model = predict.load_saved_model_guided(model_path=model_path)

    if dataset == 'All':
        for dataset in ['mudanjiang', 'yidayi', 'xwzc', 'four_center_data']:
            predict_all_chinese_dataset(simulation_only, fold, dataset)
        return None

    from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_simulate_clot.form_raw_dataset import \
        get_top_dicts as get_top_dicts_rescaled_ct_and_depth

    top_dict_ct, top_dict_depth_and_branch = get_top_dicts_rescaled_ct_and_depth(dataset, denoise=False)

    list_file_name = os.listdir(top_dict_ct)[fold[0]::fold[1]]

    name_qualified_scans = os.listdir('/data_disk/pulmonary_embolism_final/training_samples_simulate_clot/'
                                      'high_resolution/not_pe_ready_not_denoise')
    processed = 0
    for scan_name in list_file_name:
        print("processing:", scan_name, processed, '/', len(list_file_name))
        save_path = os.path.join(save_top_dict, 'clot_av_chinese', scan_name[:-4] + '.pickle')
        if os.path.exists(save_path):
            print(scan_name, "processed")
            processed += 1
            continue
        if not scan_name[:-4] + '.pickle' in name_qualified_scans:
            print(scan_name, "not qualified")
            processed += 1
            continue

        rescaled_ct = np.load(os.path.join(top_dict_ct, scan_name))['array']

        path_depth_array = os.path.join(top_dict_depth_and_branch, 'depth_array', scan_name)
        depth_array = np.load(path_depth_array)['array']

        path_branch_array = os.path.join(top_dict_depth_and_branch, 'blood_branch_map', scan_name)
        branch_array = np.load(path_branch_array)['array']

        top_dict_semantic = top_dict_depth_and_branch.replace('depth_and_center-line', 'semantics')
        artery_path = os.path.join(top_dict_semantic, 'artery_mask', scan_name)
        artery_mask = np.load(artery_path)['array']
        vein_path = os.path.join(top_dict_semantic, 'vein_mask', scan_name)
        vein_mask = np.load(vein_path)['array']

        top_dict_secondary_semantic = top_dict_depth_and_branch.replace('depth_and_center-line', 'secondary_semantics')
        path_blood_region_strict = os.path.join(top_dict_secondary_semantic, 'blood_region_strict', scan_name)
        blood_region_strict = np.load(path_blood_region_strict)['array']

        metrics = process_to_get_metrics(rescaled_ct, depth_array, artery_mask, vein_mask, branch_array,
                                         blood_region_strict, model)

        Functions.pickle_save_object(save_path, metrics)
        processed += 1


def get_prediction_clot_rad(scan_name, top_dict_dataset='/data_disk/RAD-ChestCT_dataset', model=None,
                            top_dict_save=None):
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    save_path = os.path.join(top_dict_save, 'clot_av_rad', scan_name[:-4] + '.pickle')
    if os.path.exists(save_path):
        print(scan_name, "processed")
        return None

    path_rescaled_ct = os.path.join(top_dict_dataset, 'stack_ct_rad_format', scan_name)
    rescaled_ct = load_func_for_ct(path_rescaled_ct)

    depth_array = np.load(
        os.path.join(top_dict_dataset, 'depth_and_center-line/depth_array', scan_name))['array']
    branch_array = np.load(
        os.path.join(top_dict_dataset, 'depth_and_center-line/blood_branch_map', scan_name))['array']

    artery_mask = np.load(
        os.path.join(top_dict_dataset, 'semantics/artery_mask', scan_name))['array']

    vein_mask = np.load(
        os.path.join(top_dict_dataset, 'semantics/vein_mask', scan_name))['array']

    blood_region_strict = np.load(
        os.path.join(top_dict_dataset, 'secondary_semantics/blood_region_strict', scan_name))['array']

    metrics = process_to_get_metrics(rescaled_ct, depth_array, artery_mask, vein_mask, branch_array,
                                     blood_region_strict, model)

    Functions.pickle_save_object(save_path, metrics)


def temp_func_for_meeting(rescaled_ct, depth_array, artery_mask, vein_mask, branch_array,
                           blood_region_strict, model, return_sequence=False):
    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, depth_array, branch_array, absolute_cube_length=absolute_cube_length,
        exclude_center_out=exclude_center_out, min_depth=min_depth_get_sequence)

    # here the guided mask is use the blood vessel mask (same with training condition)
    sample_sequence_v0 = predict.predict_clot_for_sample_sequence(
        sample_sequence, model=model, min_depth=0.5, trim=True, trim_length=trim_length)

    return sample_sequence_v0


def process_to_get_metrics(rescaled_ct, depth_array, artery_mask, vein_mask, branch_array,
                           blood_region_strict, model, return_sequence=False):
    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, depth_array, branch_array, absolute_cube_length=absolute_cube_length,
        exclude_center_out=exclude_center_out, min_depth=min_depth_get_sequence)

    # here the guided mask is use the blood vessel mask (same with training condition)
    sample_sequence_v0 = predict.predict_clot_for_sample_sequence(
        sample_sequence, model=model, min_depth=0.5, trim=True, trim_length=trim_length)

    clot_prob_v0 = reconstruct_semantic_from_sample_sequence(
        sample_sequence_v0, absolute_cube_length, key='clot_prob_mask', background=0)
    print("v0: blood vessel as guide")
    v0, v0_strict, _ = analysis_clot_in_av(clot_prob_v0, artery_mask, vein_mask, blood_region_strict)
    metrics = {"v0": v0, "v0_strict": v0_strict}

    return metrics, sample_sequence_v0


def get_prediction_clot_paired_dataset(scan_name, top_dict_dataset='/data_disk/CTA-CT_paired-dataset',
                                       model=None, top_dict_save=None):

    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    if scan_name in ['zryh-0037.npz', ]:
        print("wrong file")
        return None

    save_path = os.path.join(top_dict_save, 'clot_av_paired_dataset', scan_name[:-4] + '.pickle')
    if os.path.exists(save_path):
        print(scan_name, "processed")
        return None

    dataset_dict_cta, dataset_dict_non = basic_functions.find_patient_id_dataset_correspondence(
        scan_name, top_dict=top_dict_dataset, strip=True)

    print(scan_name, 'in', dataset_dict_non)

    rescaled_ct = np.load(os.path.join(dataset_dict_non, 'rescaled_ct', scan_name))['array']

    depth_array = np.load(
        os.path.join(dataset_dict_non, 'depth_and_center-line/depth_array', scan_name))['array']
    branch_array = np.load(
        os.path.join(dataset_dict_non, 'depth_and_center-line/blood_branch_map', scan_name))['array']
    artery_mask = np.load(
        os.path.join(dataset_dict_non, 'semantics/artery_mask', scan_name))['array']
    vein_mask = np.load(
        os.path.join(dataset_dict_non, 'semantics/vein_mask', scan_name))['array']
    blood_region_strict = np.load(
        os.path.join(dataset_dict_non, 'secondary_semantics/blood_region_strict', scan_name))['array']

    metrics = process_to_get_metrics(rescaled_ct, depth_array, artery_mask, vein_mask, branch_array,
                                     blood_region_strict, model)

    Functions.pickle_save_object(save_path, metrics)


def analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, blood_region_strict=None):

    artery_volume = np.sum(artery_mask) + 0.001
    vein_volume = np.sum(vein_mask) + 0.001

    clot_in_artery = predict_clot_mask * artery_mask
    clot_in_vein = predict_clot_mask * vein_mask

    ratio_clot_artery = np.sum(clot_in_artery) / artery_volume
    total_artery_clot_volume = ratio_clot_artery * artery_volume * (334 / 512 * 334 / 512)
    ratio_clot_vein = np.sum(clot_in_vein) / vein_volume

    print("a-v clot ratio:", ratio_clot_artery / ratio_clot_vein, "   artery clot ratio:", ratio_clot_artery,
          "   artery clot volume in mm^3", total_artery_clot_volume)

    if blood_region_strict is None:
        return ratio_clot_artery / ratio_clot_vein, ratio_clot_artery, total_artery_clot_volume

    artery_strict = blood_region_strict * artery_mask
    vein_strict = blood_region_strict * vein_mask

    artery_volume_strict = np.sum(artery_strict) + 0.001
    vein_volume_strict = np.sum(vein_strict) + 0.001

    clot_in_artery_strict = predict_clot_mask * artery_strict
    clot_in_vein_strict = predict_clot_mask * vein_strict

    ratio_clot_artery_strict = np.sum(clot_in_artery_strict) / artery_volume_strict
    total_artery_clot_volume_strict = ratio_clot_artery_strict * artery_volume_strict * (334 / 512 * 334 / 512)
    ratio_clot_vein_strict = np.sum(clot_in_vein_strict) / vein_volume_strict

    print("a-v clot ratio strict:", ratio_clot_artery_strict / ratio_clot_vein_strict,
          "   artery clot strict:", ratio_clot_artery_strict,
          "   artery clot volume strict in mm^3", total_artery_clot_volume_strict)

    return (ratio_clot_artery / ratio_clot_vein, ratio_clot_artery, total_artery_clot_volume), \
           (ratio_clot_artery_strict / ratio_clot_vein_strict, ratio_clot_artery_strict,
            total_artery_clot_volume_strict)


if __name__ == '__main__':
    Functions.set_visible_device('0')
    current_fold = (0, 2)
    predict_all_chinese_dataset(fold=current_fold, simulation_only=False)
    predict_all_chinese_dataset(fold=current_fold, simulation_only=True)

    exit()
    predict_all_rad(fold=current_fold, simulation_only=True)
    predict_all_paired_dataset(fold=current_fold, simulation_only=True)
    predict_all_rad(fold=current_fold, simulation_only=False)
    predict_all_paired_dataset(fold=current_fold, simulation_only=False)
    exit()
