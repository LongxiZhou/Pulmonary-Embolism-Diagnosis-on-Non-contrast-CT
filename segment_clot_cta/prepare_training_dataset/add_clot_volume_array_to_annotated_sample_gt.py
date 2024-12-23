"""
clot_volume_array is used to form penalty for clots.
see segment_clot_cta.prepare_training_dataset.simulate_clot_pe_v3.assign_lesion_volume_array

for annotated gt, we directly set the clot volume to very small, thus, model will focus more on annotated clots
"""


import os
import Tool_Functions.Functions as Functions
import numpy as np
from format_convert.spatial_normalize import rescale_to_new_shape


def trim_blood_and_clot_and_assign_clot_volume_array_gt(sample, denoise=False, clot_volume=50):
    assert sample['is_PE_and_has_clot_gt']

    # clot attention is sqrt(2000 / 50) times higher than simulated clots
    clot_volume = clot_volume * np.sqrt(sample['clot_volume_sum'] / 2000)

    sample_sequence_with_clot = sample['sample_sequence']

    if denoise:
        baseline_ct = Functions.change_to_rescaled(-100)
    else:
        baseline_ct = Functions.change_to_rescaled(-200)

    for item in sample_sequence_with_clot:

        ct_data = item['ct_data']
        valid_mask = np.array(ct_data > baseline_ct, 'float16')

        if item['depth_cube'] is not None:
            item['depth_cube'] = item['depth_cube'] * valid_mask

        if item['clot_array'] is not None:
            if not np.shape(item['clot_array']) == np.shape(ct_data):
                item['clot_array'] = np.array(rescale_to_new_shape(item['clot_array'], np.shape(ct_data)), 'float16')
            item['clot_array'] = item['clot_array'] * valid_mask
            clot_volume_array = \
                np.array(item['clot_array'] > 0, 'float16') + np.array(item['clot_array'] < 0, 'float16')
            clot_volume_array = clot_volume_array * clot_volume
        else:
            clot_volume_array = None

        if item['blood_region'] is not None:
            item['blood_region'] = item['blood_region'] * valid_mask

        item['clot_volume_array'] = clot_volume_array

    sample['sample_sequence'] = sample_sequence_with_clot

    return sample


def process_dataset(top_dict_dataset, save_dict_dataset=None, denoise=False, clot_volume=50, fold=(0, 1)):
    if save_dict_dataset is None:
        save_dict_dataset = top_dict_dataset

    fn_list = os.listdir(top_dict_dataset)[fold[0]::fold[1]]

    for fn in Functions.iteration_with_time_bar(fn_list):
        path_load = os.path.join(top_dict_dataset, fn)
        path_save = os.path.join(save_dict_dataset, fn)

        sample = Functions.pickle_load_object(path_load)

        sample_processed = trim_blood_and_clot_and_assign_clot_volume_array_gt(
            sample, denoise=denoise, clot_volume=clot_volume)

        Functions.pickle_save_object(path_save, sample_processed)


if __name__ == '__main__':
    current_fold = (0, 1)
    top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/loop_3_59/' \
               'pe_v3_long_length_complete_vessel/'

    process_dataset(top_dict + 'denoise_low-resolution', denoise=True, clot_volume=50, fold=current_fold)
    process_dataset(top_dict + 'original_low-resolution', denoise=False, clot_volume=50, fold=current_fold)
    exit()

    top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/initial_50/' \
               'pe_v3_long_length_complete_vessel/'

    process_dataset(top_dict + 'denoise_low-resolution', denoise=True, clot_volume=50, fold=current_fold)
    process_dataset(top_dict + 'original_low-resolution', denoise=False, clot_volume=50, fold=current_fold)

    top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/next_82/' \
               'pe_v3_long_length_complete_vessel/'

    process_dataset(top_dict + 'denoise_low-resolution', denoise=True, clot_volume=50, fold=current_fold)
    process_dataset(top_dict + 'original_low-resolution', denoise=False, clot_volume=50, fold=current_fold)
    exit()

    top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/sample_sequence/' \
               'pe_v3/'

    process_dataset(top_dict + 'denoise_high-resolution', denoise=True, clot_volume=300)
    process_dataset(top_dict + 'denoise_low-resolution', denoise=True, clot_volume=50)
    process_dataset(top_dict + 'original_high-resolution', denoise=False, clot_volume=300)
    process_dataset(top_dict + 'original_low-resolution', denoise=False, clot_volume=50)
    exit()

    from segment_clot_cta.prepare_training_dataset.simulate_clot_pe_v3 import visualize_clots_on_sample_sequence

    sample_ = Functions.pickle_load_object('/data_disk/pulmonary_embolism/segment_clot_on_CTA/'
                                           'PE_CTA_with_gt/sample_sequence/patient-id-017.pickle')
    sample_modified = trim_blood_and_clot_and_assign_clot_volume_array_gt(sample_)

    visualize_clots_on_sample_sequence(sample_modified['sample_sequence'], high_resolution=False, clip_window=True,
                                       z_start=278)
    exit()

    Functions.pickle_save_object('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/sample_sequence/'
                                 'patient-id-017_modified.pickle', sample_modified)
