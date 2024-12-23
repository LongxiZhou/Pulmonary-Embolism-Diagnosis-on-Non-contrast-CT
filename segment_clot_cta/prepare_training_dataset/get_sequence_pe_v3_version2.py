"""

"""


def form_sample_sequence_dataset_use_simulated_non_contrast(denoise=False, high_resolution=True, fold=(0, 1),
                                                            save_dict=None, dataset_dict=None, strict_trim=True,
                                                            exclude_center_out=False, gt_only=False, wrong_list=None):
    # semantics is calculated based on simulated non contrast
    # sample_sequence is trimmed and ready for inference.
    global top_dict_dataset, sample_sequence_save_dict

    if save_dict is not None:
        sample_sequence_save_dict = save_dict
    if dataset_dict is not None:
        top_dict_dataset = dataset_dict

    if denoise and high_resolution:
        folder_name = 'denoise_high-resolution'
    elif denoise and not high_resolution:
        folder_name = 'denoise_low-resolution'
    elif not denoise and high_resolution:
        folder_name = 'original_high-resolution'
    else:  # not denoise and not high_resolution:
        folder_name = 'original_low-resolution'

    current_sample_sequence_save_dict = sample_sequence_save_dict + folder_name + '/'

    from pulmonary_embolism_v3.prepare_training_dataset.convert_ct_to_sample import pipeline_process_adaptive_new

    pipeline_process_adaptive_new(denoise=denoise, only_v1=True, high_resolution=high_resolution, fold=fold,
                                  wrong_list=wrong_list, top_dict_source=top_dict_dataset,
                                  top_dict_sample_sequence=current_sample_sequence_save_dict, strict_trim=strict_trim,
                                  exclude_center_out=exclude_center_out, gt_only=gt_only)


def add_sample_sequence_for_new_annotations_paired_dataset():
    import os
    import pe_dataset_management.basic_functions as basic_functions

    current_fold = (0, 1)
    global top_dict_dataset, sample_sequence_save_dict

    #################
    # Note: for ground truth annotation, further run
    # segment_clot_cta.prepare_training_dataset.add_clot_volume_array_to_annotated_sample_gt.py
    #################

    top_dict_database = '/data_disk/CTA-CT_paired-dataset/dataset_CTA/'

    sample_sequence_save_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/' \
                                'loop_3_59/pe_v3_long_length_complete_vessel/'

    # sample sequence processed in the previous loops
    wrong_name_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt_82/clot_gt')
    wrong_name_list = wrong_name_list + os.listdir(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt_50/clot_gt')

    trim_strict = False  # if True, trim high(low) resolution to 3000(1500); otherwise 4000(3000)
    exclude_center_out_ = False  # if True, only cube center in the mask will be included
    # True to exclude cubes if its center is outside the valid_mask, i.e., vessel mask
    # False, all valid mask will be included in the sample_sequence

    gt_only_ = True  # will only process file name in os.path.join(top_dict_dataset, 'clot_gt')

    pe_dataset = basic_functions.get_dataset_relative_path(scan_class='PE')
    for dataset in pe_dataset:
        top_dict_dataset = os.path.join(top_dict_database, dataset)
        print("processing:", top_dict_dataset)

        form_sample_sequence_dataset_use_simulated_non_contrast(
            denoise=True, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
            exclude_center_out=exclude_center_out_, gt_only=gt_only_, wrong_list=wrong_name_list)
        form_sample_sequence_dataset_use_simulated_non_contrast(
            denoise=False, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
            exclude_center_out=exclude_center_out_, gt_only=gt_only_, wrong_list=wrong_name_list)

    exit()


def add_sample_sequence_without_annotations_paired_dataset():
    import os
    import pe_dataset_management.basic_functions as basic_functions
    import Tool_Functions.file_operations as file_operations

    current_fold = (0, 1)
    global top_dict_dataset, sample_sequence_save_dict

    #################
    # Note: for ground truth annotation, further run
    # segment_clot_cta.prepare_training_dataset.add_clot_volume_array_to_annotated_sample_gt.py
    #################

    top_dict_database = '/data_disk/CTA-CT_paired-dataset/dataset_CTA/'

    sample_sequence_save_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_no_gt/' \
                                'pe_v3_long_length_complete_vessel/'

    # sample sequence processed in the previous loops
    sample_with_gt_list = file_operations.extract_all_file_path(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt', end_with='.pickle')

    annotated_name_list = set()
    for path in sample_with_gt_list:
        file_name = path.split('/')[-1][:-7] + '.npz'
        annotated_name_list.add(file_name)

    annotated_name_list = list(annotated_name_list)
    print(len(annotated_name_list), annotated_name_list)

    trim_strict = False  # if True, trim high(low) resolution to 3000(1500); otherwise 4000(3000)
    exclude_center_out_ = False  # if True, only cube center in the mask will be included
    # True to exclude cubes if its center is outside the valid_mask, i.e., vessel mask
    # False, all valid mask will be included in the sample_sequence

    gt_only_ = False  # will only process file name in os.path.join(top_dict_dataset, 'clot_gt')

    pe_dataset = basic_functions.get_dataset_relative_path(scan_class='PE')
    for dataset in pe_dataset:
        top_dict_dataset = os.path.join(top_dict_database, dataset)
        print("processing:", top_dict_dataset)

        form_sample_sequence_dataset_use_simulated_non_contrast(
            denoise=True, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
            exclude_center_out=exclude_center_out_, gt_only=gt_only_, wrong_list=annotated_name_list)
        form_sample_sequence_dataset_use_simulated_non_contrast(
            denoise=False, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
            exclude_center_out=exclude_center_out_, gt_only=gt_only_, wrong_list=annotated_name_list)

    exit()


if __name__ == '__main__':
    add_sample_sequence_without_annotations_paired_dataset()
    exit()
