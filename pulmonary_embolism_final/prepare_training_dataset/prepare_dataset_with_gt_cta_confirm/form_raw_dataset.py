from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import convert_ct_into_tubes
import Tool_Functions.Functions as Functions
import pe_dataset_management.basic_functions as basic_functions
import numpy as np
import pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.get_clot_gt_and_penalty as \
    get_clot_gt_and_penalty
from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.stratify_gt_quality import \
    get_quality_of_scan_name
import os


def process_scan_list(fn_list, high_resolution=True, fold=(0, 1),
                      denoise=False, vessel_high_recall=True, for_evaluation=False, parameter_dict=None):

    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
        min_depth = 2.5
        exclude_center_out = False
    else:
        absolute_cube_length = (4, 4, 5)
        min_depth = 0.5
        exclude_center_out = True

    if vessel_high_recall:
        top_dict_save = '/data_disk/pulmonary_embolism_final/' \
                        'training_samples_with_annotation_vessel_high_recall_cta_confirm/'
    else:
        top_dict_save = '/data_disk/pulmonary_embolism_final/training_samples_with_annotation_cta_confirm/'

    if for_evaluation:
        if vessel_high_recall:
            top_dict_save = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation_cta_confirm/' \
                            'pe_vessel_high_recall/'
        else:
            top_dict_save = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation_cta_confirm/' \
                            'pe/'

    if high_resolution:
        top_dict_save = os.path.join(top_dict_save, 'high_resolution')
    else:
        top_dict_save = os.path.join(top_dict_save, 'low_resolution')

    if denoise:
        save_dict_dataset = os.path.join(top_dict_save, 'pe_not_trim_denoise')
    else:
        save_dict_dataset = os.path.join(top_dict_save, 'pe_not_trim_not_denoise')

    fn_list = list(fn_list)
    fn_list = Functions.split_list_by_ord_sum(fn_list, fold=fold)
    processed_count = 0
    for file_name in fn_list:
        if len(file_name) <= 4:
            file_name = file_name + '.npz'
        if len(file_name) > 4:
            if not file_name[-4:] == '.npz':
                file_name = file_name + '.npz'
        print("\nprocessing:", file_name, len(fn_list) - processed_count, 'left')
        if os.path.exists(os.path.join(save_dict_dataset, file_name[:-4] + '.pickle')):
            print('processed')
            processed_count += 1
            continue

        fp_penalty_array, clot_gt_probability_non = \
            get_clot_gt_and_penalty.derive_penalty_array_and_clot_gt_non_from_clot_segmentation_on_cta(
                file_name, extra_layer=2)
        dataset_cta, dataset_non = basic_functions.find_patient_id_dataset_correspondence(file_name, strip=True)

        if denoise:
            rescaled_ct = np.load(os.path.join(dataset_non, 'rescaled_ct', file_name))['array']
        else:
            rescaled_ct = np.load(os.path.join(dataset_non, 'rescaled_ct-denoise', file_name))['array']

        top_dict_depth_and_branch = os.path.join(dataset_non, 'depth_and_center-line')

        if vessel_high_recall:
            path_depth_array = os.path.join(
                top_dict_depth_and_branch, 'high_recall_depth_array', file_name[:-4] + '.npz')
        else:
            path_depth_array = os.path.join(top_dict_depth_and_branch, 'depth_array', file_name[:-4] + '.npz')
        depth_array = np.load(path_depth_array)['array']

        if vessel_high_recall:
            path_branch_array = os.path.join(
               top_dict_depth_and_branch, 'high_recall_blood_branch_map', file_name[:-4] + '.npz')
        else:
            path_branch_array = os.path.join(top_dict_depth_and_branch, 'blood_branch_map', file_name[:-4] + '.npz')
        branch_array = np.load(path_branch_array)['array']

        if vessel_high_recall:
            blood_center_line_path = os.path.join(
                top_dict_depth_and_branch, 'blood_high_recall_center_line', file_name[:-4] + '.npz')
        else:
            blood_center_line_path = os.path.join(
                top_dict_depth_and_branch, 'blood_center_line', file_name[:-4] + '.npz')
        center_line_mask = np.load(blood_center_line_path)['array']

        if for_evaluation:
            top_dict_semantic = top_dict_depth_and_branch.replace('depth_and_center-line', 'semantics')
            artery_path = os.path.join(top_dict_semantic, 'artery_mask', file_name[:-4] + '.npz')
            artery_mask = np.load(artery_path)['array']
            vein_path = os.path.join(top_dict_semantic, 'vein_mask', file_name[:-4] + '.npz')
            vein_mask = np.load(vein_path)['array']

            semantic_dict = {"artery_mask": artery_mask, "vein_mask": vein_mask}
        else:
            semantic_dict = {}

        sample_list = convert_ct_into_tubes(
            rescaled_ct, depth_array, branch_array, absolute_cube_length=absolute_cube_length, min_depth=min_depth,
            exclude_center_out=exclude_center_out, penalty_weight_fp=fp_penalty_array,
            clot_gt_mask=clot_gt_probability_non, **semantic_dict)

        print("sample list has:", len(sample_list), "elements")

        center_line_loc_array = np.where(center_line_mask > 0.5)

        sample_final = {"sample_sequence": sample_list, "center_line_loc_array": center_line_loc_array,
                        "clot_gt_volume_sum": np.sum(clot_gt_probability_non)}
        if parameter_dict is not None:
            key_set = parameter_dict.keys()
            for key in key_set:
                sample_final[key] = parameter_dict[key]

        save_path = os.path.join(save_dict_dataset, file_name[:-4] + '.pickle')
        print("saving to:", save_path)
        Functions.pickle_save_object(save_path, sample_final)
        processed_count += 1


def pipeline_process_human_annotation(high_resolution=True, fold=(0, 1), denoise=False,
                                      vessel_high_recall=True, for_evaluation=False):
    fn_good_pair_good_registration, fn_good_pair_excellent_registration, \
        fn_excellent_pair_good_registration, fn_excellent_pair_excellent_registration = get_quality_of_scan_name()

    dict_a = {"is_PE": True, "has_clot_gt": True, "human_annotation": True,
              "registration_quality": "perfect", "pe_pair_quality": "perfect"}
    process_scan_list(fn_excellent_pair_excellent_registration, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation,
                      parameter_dict=dict_a)

    dict_b = {"is_PE": True, "has_clot_gt": True, "human_annotation": True,
              "registration_quality": "perfect", "pe_pair_quality": "good"}
    process_scan_list(fn_good_pair_excellent_registration, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation,
                      parameter_dict=dict_b)

    dict_c = {"is_PE": True, "has_clot_gt": True, "human_annotation": True,
              "registration_quality": "good", "pe_pair_quality": "perfect"}
    process_scan_list(fn_excellent_pair_good_registration, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation,
                      parameter_dict=dict_c)

    dict_d = {"is_PE": True, "has_clot_gt": True, "human_annotation": True,
              "registration_quality": "good", "pe_pair_quality": "good"}
    process_scan_list(fn_good_pair_good_registration, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation,
                      parameter_dict=dict_d)


def pipeline_high_quality_not_human_annotation(high_resolution=True, fold=(0, 1), denoise=False,
                                               vessel_high_recall=True, for_evaluation=False):
    with_annotation = set()
    for name_set in get_quality_of_scan_name():
        with_annotation = with_annotation | name_set
    scan_name_pe = basic_functions.get_all_scan_name(scan_class='PE', dir_key_word='High_Quality')
    scan_name_temp = basic_functions.get_all_scan_name(scan_class='Temp', dir_key_word='High_Quality')

    scan_name_not_annotated = (set(scan_name_pe) | set(scan_name_temp)) - with_annotation

    dict_d = {"has_clot_gt": True, "human_annotation": False,
              "registration_quality": "unknown", "pe_pair_quality": "perfect"}

    process_scan_list(scan_name_not_annotated, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation,
                      parameter_dict=dict_d)


# only for evaluation, will not include in training
def pipeline_other_scans(high_resolution=True, fold=(0, 1), denoise=False, vessel_high_recall=True):
    scan_name_pe_low = basic_functions.get_all_scan_name(scan_class='PE', dir_exclusion_key_word='High_Quality')
    scan_name_temp_low = basic_functions.get_all_scan_name(scan_class='Temp', dir_exclusion_key_word='High_Quality')
    scan_name_normal = basic_functions.get_all_scan_name(scan_class='Normal', dir_exclusion_key_word='High_Quality')

    dict_e = {"has_clot_gt": False, "human_annotation": False,
              "registration_quality": "unknown", "pe_pair_quality": "low"}

    scan_list = scan_name_normal + scan_name_pe_low + scan_name_temp_low

    process_scan_list(scan_list, high_resolution=high_resolution,
                      fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall,
                      for_evaluation=True, parameter_dict=dict_e)


def process_all(fold=(0, 1)):
    pipeline_other_scans(high_resolution=True, denoise=False, vessel_high_recall=True, fold=fold)
    for evaluation in [False, True]:
        pipeline_process_human_annotation(high_resolution=True, denoise=False, vessel_high_recall=True, fold=fold,
                                          for_evaluation=evaluation)
        pipeline_high_quality_not_human_annotation(high_resolution=True, denoise=False, vessel_high_recall=True,
                                                   fold=fold, for_evaluation=evaluation)


if __name__ == '__main__':
    current_fold = (0, 12)
    Functions.set_visible_device('1')
    process_all(current_fold)
    exit()
