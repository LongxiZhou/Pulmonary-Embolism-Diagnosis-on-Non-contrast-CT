"""

"""


def form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=(0, 1), save_dict=None,
                                 dataset_dict=None, strict_trim=True, exclude_center_out=False):

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

    from pulmonary_embolism_v3.prepare_training_dataset.convert_ct_to_sample import pipeline_process_adaptive

    pipeline_process_adaptive(denoise=denoise, only_v1=True, high_resolution=high_resolution, fold=fold,
                              wrong_list=None, top_dict_source=top_dict_dataset,
                              top_dict_sample_sequence=current_sample_sequence_save_dict, strict_trim=strict_trim,
                              exclude_center_out=exclude_center_out)


if __name__ == '__main__':
    current_fold = (0, 2)
    top_dict_dataset = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/non_PE_CTA/'
    sample_sequence_save_dict = top_dict_dataset + 'sample_sequence/pe_v3_long_length_complete_vessel/'
    trim_strict = False  # if True, trim high(low) resolution to 3000(1500); otherwise 4000(3000)
    exclude_center_out_ = False  # if True, only cube center in the mask will be included
    # True to exclude cubes if its center is outside the valid_mask, i.e., vessel mask
    # False, all valid mask will be included in the sample_sequence

    form_sample_sequence_dataset(denoise=True, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
                                 exclude_center_out=exclude_center_out_)
    form_sample_sequence_dataset(denoise=False, high_resolution=False, fold=current_fold, strict_trim=trim_strict,
                                 exclude_center_out=exclude_center_out_)

    exit()

    form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=current_fold, strict_trim=trim_strict,
                                 exclude_center_out=exclude_center_out_)
    form_sample_sequence_dataset(denoise=True, high_resolution=True, fold=current_fold, strict_trim=trim_strict,
                                 exclude_center_out=exclude_center_out_)
