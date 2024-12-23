"""
This dataset if for refining the model, and let the model know how blood clot looks like in non-contrast CT.

raw dataset is from non-contrast CT in CTA-non pairs.
CTA confirmed PE positive, and non-contrast is collected not so long before CTA
Thus, all these non-contrast CT should have very high probability of PE

remove the sample if:
it is not non-contrast (blood signal not in [0, 100])
or we are not sure it is PE positive

remove some patch of the sample sequence to save GPU memory and control variance
remove patch if:
the branch level > 7
the length is > 1500 for low resolution and > 4000 for high resolution  (first remove patch with high branch level)
"""


from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.trim_and_remove_bad_scan_pe import \
    trim_and_reduce_for_dataset
import os
import pulmonary_embolism_final.prepare_training_dataset.\
    prepare_dataset_with_gt_cta_confirm.scan_name_diagnosis_type as diagnosis_type
import Tool_Functions.Functions as Functions


def general_refine_settings(high_resolution):
    if not high_resolution:
        target_length = 1500
        max_branch = 9
    else:
        target_length = 4000
        max_branch = 9

    return target_length, max_branch


def refine_dataset(top_dict='/data_disk/pulmonary_embolism_final/training_samples_with_annotation_'
                            'vessel_high_recall_cta_confirm',
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


def exclude_not_sure_samples_and_assign_relative_importance(
        sample_dir='/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_'
                   'high_recall_cta_confirm/high_resolution/pe_not_trim_not_denoise'):

    # only call once

    pe_name_relative_importance_list = diagnosis_type.get_pe_scan_name_and_relative_importance()

    name_set_cta_confirm = set()
    for name, importance in pe_name_relative_importance_list:
        name_set_cta_confirm.add(name)

    sample_name_list = os.listdir(sample_dir)
    for sample_name in sample_name_list:
        if not sample_name[:-7] in name_set_cta_confirm:
            path_remove = os.path.join(sample_dir, sample_name)
            if os.path.exists(path_remove):
                print("removing:", path_remove)
                os.remove(path_remove)

    for name, importance in pe_name_relative_importance_list:
        path_sample = os.path.join(sample_dir, name + '.pickle')
        if os.path.exists(path_sample):
            sample = Functions.pickle_load_object(path_sample)
            if "relative_importance" in sample.keys():
                raise ValueError("seems already assigned")
            sample["relative_importance"] = importance
            Functions.pickle_save_object(path_sample, sample)


if __name__ == '__main__':
    # exclude_not_sure_samples_and_assign_relative_importance()
    refine_dataset(fold=(0, 8))
