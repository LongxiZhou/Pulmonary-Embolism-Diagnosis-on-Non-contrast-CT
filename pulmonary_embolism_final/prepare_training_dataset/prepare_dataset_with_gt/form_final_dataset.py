"""
This dataset if for refining the model, and let the model know how blood clot looks like in non-contrast CT.

raw dataset is from non-contrast CT in CTA-non pairs.
CTA confirmed PE positive, and non-contrast is collected not so long before CTA
Thus, all these non-contrast CT should have very high probability of PE

remove the sample if:
it is not non-contrast (blood signal not in [0, 100])

remove some patch of the sample sequence to save GPU memory and control variance
remove patch if:
the branch level > 7
the length is > 1500 for low resolution and > 4000 for high resolution  (first remove patch with high branch level)
"""


from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.trim_and_remove_bad_scan_pe import \
    trim_and_reduce_for_dataset
import os


def general_refine_settings(high_resolution):
    if not high_resolution:
        target_length = 1500
        max_branch = 7
    else:
        target_length = 4000
        max_branch = 7

    return target_length, max_branch


def refine_dataset(top_dict='/data_disk/pulmonary_embolism_final/training_samples_with_annotation',
                   high_resolution=True, fold=(0, 1)):
    target_length, max_branch = general_refine_settings(high_resolution)

    if high_resolution:
        top_dict = os.path.join(top_dict, 'high_resolution')
    else:
        top_dict = os.path.join(top_dict, 'low_resolution')

    folder_raw_dataset = ['pe_not_trim_denoise', 'pe_not_trim_not_denoise']
    folder_refined_dataset = ['pe_ready_denoise', 'pe_ready_not_denoise']

    for index in range(len(folder_raw_dataset)):
        dir_raw = os.path.join(top_dict, folder_raw_dataset[index])
        dir_refined = os.path.join(top_dict, folder_refined_dataset[index])

        trim_and_reduce_for_dataset(
            dir_raw, dir_refined, high_resolution=high_resolution, fold=fold, reprocess=False,
            target_length=target_length, max_branch=max_branch, cta=False)


def refine_all(fold=(0, 1)):
    refine_dataset('/data_disk/pulmonary_embolism_final/training_samples_with_annotation', True, fold)
    refine_dataset('/data_disk/pulmonary_embolism_final/training_samples_with_annotation', False, fold)
    refine_dataset('/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_high_recall',
                   True, fold)
    refine_dataset('/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_high_recall',
                   False, fold)


if __name__ == '__main__':
    current_fold = (0, 1)
    refine_all(fold=current_fold)
