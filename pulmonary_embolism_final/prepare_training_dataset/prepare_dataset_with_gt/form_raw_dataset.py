from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import convert_ct_into_tubes
import Tool_Functions.Functions as Functions
import pe_dataset_management.basic_functions as basic_functions
import numpy as np
import pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.get_clot_gt_and_penalty as \
    get_clot_gt_and_penalty
from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.stratify_gt_quality import \
    get_quality_of_scan_name
import os


def process_pe(fn_list, registration_quality, pe_pair_quality, high_resolution=False, fold=(0, 1),
               denoise=True, vessel_high_recall=False, for_evaluation=False):

    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
        min_depth = 2.5
        exclude_center_out = False
    else:
        absolute_cube_length = (4, 4, 5)
        min_depth = 0.5
        exclude_center_out = True

    if vessel_high_recall:
        top_dict_save = '/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_high_recall/'
    else:
        top_dict_save = '/data_disk/pulmonary_embolism_final/training_samples_with_annotation/'

    if for_evaluation:
        if vessel_high_recall:
            top_dict_save = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation/' \
                            'pe_vessel_high_recall/'
        else:
            top_dict_save = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation/' \
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
    fn_list.sort()
    fn_list = fn_list[fold[0]::fold[1]]
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

        fp_penalty_array, clot_gt_probability_non = \
            get_clot_gt_and_penalty.derive_penalty_array_and_clot_gt_non_from_clot_segmentation_on_cta(file_name)

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
                        "is_PE": True, "has_clot_gt": True, "clot_gt_volume_sum": np.sum(clot_gt_probability_non),
                        "registration_quality": registration_quality, "pe_pair_quality": pe_pair_quality}

        Functions.pickle_save_object(os.path.join(save_dict_dataset, file_name[:-4] + '.pickle'), sample_final)
        processed_count += 1


def pipeline_process(high_resolution=False, fold=(0, 1), denoise=True, vessel_high_recall=True, for_evaluation=False):
    Functions.set_visible_device(str(fold[0] % 2 + 1))
    fn_good_pair_good_registration, fn_good_pair_excellent_registration, \
        fn_excellent_pair_good_registration, fn_excellent_pair_excellent_registration = get_quality_of_scan_name()

    process_pe(fn_excellent_pair_excellent_registration, registration_quality="perfect",
               pe_pair_quality="perfect", high_resolution=high_resolution,
               fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation)
    process_pe(fn_good_pair_excellent_registration, registration_quality="perfect",
               pe_pair_quality="good", high_resolution=high_resolution,
               fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation)
    process_pe(fn_excellent_pair_good_registration, registration_quality="good",
               pe_pair_quality="perfect", high_resolution=high_resolution,
               fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation)
    process_pe(fn_good_pair_good_registration, registration_quality="good",
               pe_pair_quality="good", high_resolution=high_resolution,
               fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall, for_evaluation=for_evaluation)


def process_all(fold=(0, 1)):
    for high_resolution in [True, False]:
        for denoise in [True, False]:
            for vessel_high_recall in [True, False]:
                pipeline_process(
                    high_resolution=high_resolution, fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall)


def process_for_evaluation(fold=(0, 1)):

    # experiments did not find evidence supporting that
    # using denoise or vessel high recall will improved the performance

    for denoise in [True, False]:
        for vessel_high_recall in [True, False]:
            pipeline_process(
                high_resolution=True, fold=fold, denoise=denoise, vessel_high_recall=vessel_high_recall,
                for_evaluation=True)


if __name__ == '__main__':
    current_fold = (0, 4)
    process_for_evaluation(current_fold)
    exit()
    process_all(fold=current_fold)
    exit()
