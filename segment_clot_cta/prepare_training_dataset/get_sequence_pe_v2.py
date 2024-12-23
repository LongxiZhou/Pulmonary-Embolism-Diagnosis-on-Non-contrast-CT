"""
prepare dataset for this version:
pulmonary_embolism_v2.prepare_dataset.convert_blood_vessel_to_sliced_sequence

inference for this version:
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence
"""


def form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=(0, 1), save_dict=None,
                                 dataset_dict=None):

    # sample_sequence is trimmed and ready for inference.\
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

    from pulmonary_embolism_v2.prepare_dataset.convert_blood_vessel_to_sliced_sequence import pipeline_process

    pipeline_process(de_noise=denoise, only_v1=True, high_resolution=high_resolution, load_func_ct=None, fold=fold,
                     wrong_list=None, top_dict_source=top_dict_dataset,
                     top_dict_sample_sequence=current_sample_sequence_save_dict)


if __name__ == '__main__':
    current_fold = (0, 1)
    top_dict_dataset = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/'
    sample_sequence_save_dict = top_dict_dataset + 'sample_sequence/pe_v2/'
    form_sample_sequence_dataset(denoise=True, high_resolution=True, fold=current_fold)
    form_sample_sequence_dataset(denoise=True, high_resolution=False, fold=current_fold)
    form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=current_fold)
    form_sample_sequence_dataset(denoise=False, high_resolution=False, fold=current_fold)
