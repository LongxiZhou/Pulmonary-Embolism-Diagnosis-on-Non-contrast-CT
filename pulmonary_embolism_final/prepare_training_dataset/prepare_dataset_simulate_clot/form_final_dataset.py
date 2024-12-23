"""
this dataset is for warm-up the model and let the model understand how normal blood vessel looks like
raw dataset is from clinical non-contrast CT. the PE in these non-contrast CT should be < 0.001

remove the sample if:
it is not non-contrast (blood signal not in [0, 100])
it noise is abnormally high (noise > 150 HU in blood region)
it has too much lesion (lesion volume > 0.25 lung volume)
very bad blood vessel segmentation (length of blood vessel center line < 3500)

remove some patch of the sample sequence to save GPU memory and control variance
remove patch if:
the branch level > 7
the length is > 1500 for low resolution and > 4000 for high resolution  (first remove patch with high branch level)
"""


from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_simulate_clot.trim_and_remove_bad_scan_not_pe \
    import trim_and_reduce_for_dataset
import os


def general_refine_settings(high_resolution):
    if not high_resolution:
        target_length = 1500
        max_branch = 7
    else:
        target_length = 4000
        max_branch = 7

    return target_length, max_branch


def refine_for_high_resolution(fold=(0, 1), for_evaluation=False):
    target_length, max_branch = general_refine_settings(True)
    top_dict = '/data_disk/pulmonary_embolism_final/training_samples_simulate_clot/high_resolution'
    if for_evaluation:
        top_dict = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation/high_resolution'

    folder_raw_dataset = ['not_pe_not_trim_denoise', 'not_pe_not_trim_not_denoise']
    folder_refined_dataset = ['not_pe_ready_denoise', 'not_pe_ready_not_denoise']

    for index in range(len(folder_raw_dataset)):
        dir_raw = os.path.join(top_dict, folder_raw_dataset[index])
        dir_refined = os.path.join(top_dict, folder_refined_dataset[index])

        trim_and_reduce_for_dataset(
            dir_raw, dir_refined, high_resolution=True, fold=fold, reprocess=False,
            target_length=target_length, max_branch=max_branch, cta=False)


def refine_for_low_resolution(fold=(0, 1)):
    target_length, max_branch = general_refine_settings(False)
    top_dict = '/data_disk/pulmonary_embolism_final/training_samples_simulate_clot/low_resolution'

    folder_raw_dataset = ['not_pe_not_trim_denoise', 'not_pe_not_trim_not_denoise']
    folder_refined_dataset = ['not_pe_ready_denoise', 'not_pe_ready_not_denoise']

    for index in range(len(folder_raw_dataset)):
        dir_raw = os.path.join(top_dict, folder_raw_dataset[index])
        dir_refined = os.path.join(top_dict, folder_refined_dataset[index])

        trim_and_reduce_for_dataset(
            dir_raw, dir_refined, high_resolution=False, fold=fold, reprocess=False,
            target_length=target_length, max_branch=max_branch, cta=False)


if __name__ == '__main__':
    current_fold = (0, 4)
    refine_for_high_resolution(fold=current_fold, for_evaluation=True)
    exit()
    refine_for_high_resolution(fold=current_fold)
    refine_for_low_resolution(fold=current_fold)
