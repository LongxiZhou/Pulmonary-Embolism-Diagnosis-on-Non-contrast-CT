from chest_ct_database.feature_manager.add_blood_region import add_blood_region
from chest_ct_database.feature_manager.add_blood_region_strict import add_blood_region_strict
from chest_ct_database.feature_manager.add_depth_array import add_depth_array
from chest_ct_database.feature_manager.add_center_lines import add_center_line
from chest_ct_database.feature_manager.add_branch_map import add_branch_array
import os


def pipeline_process_all(top_dict_database, fold=(0, 1), process_av=False, only_process_av=False,
                         blood_high_recall=False):
    # for database, file structure like: top_dict_database/rescaled_ct/dataset_name
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    # top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')

    top_dict_secondary_semantics = os.path.join(top_dict_database, 'secondary_semantics')

    add_blood_region(top_dict_rescaled_ct_denoise,
                     top_dict_semantics,
                     top_dict_secondary_semantics, fold=fold,
                     process_av=process_av, only_process_av=only_process_av, blood_high_recall=blood_high_recall)

    add_depth_array(top_dict_rescaled_ct_denoise,
                    top_dict_secondary_semantics,
                    top_dict_secondary_semantics,
                    fold=fold, process_av=process_av, only_process_av=only_process_av,
                    secondary=True, blood_high_recall=blood_high_recall)

    add_center_line(top_dict_rescaled_ct_denoise,
                    top_dict_secondary_semantics,
                    top_dict_secondary_semantics,
                    fold=fold, process_av=process_av, only_process_av=only_process_av, secondary=True,
                    blood_high_recall=blood_high_recall)

    add_branch_array(top_dict_rescaled_ct_denoise,
                     top_dict_secondary_semantics,
                     top_dict_secondary_semantics,
                     fold=fold, process_av=process_av, only_process_av=only_process_av,
                     blood_high_recall=blood_high_recall)

    add_blood_region_strict(top_dict_rescaled_ct_denoise,
                            top_dict_semantics,
                            top_dict_secondary_semantics,
                            top_dict_secondary_semantics,
                            fold=fold,
                            process_av=process_av, only_process_av=only_process_av, blood_high_recall=blood_high_recall)


if __name__ == '__main__':
    current_fold = (0, 1)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/long_CTA-CT_interval/',
                         fold=current_fold,
                         process_av=True)

    exit()

    from segment_clot_cta.prepare_training_dataset.get_sequence_pe_v3 import form_sample_sequence_dataset

    top_dict_dataset = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/'
    sample_sequence_save_dict = top_dict_dataset + 'sample_sequence/pe_v3/'
    form_sample_sequence_dataset(denoise=True, high_resolution=True, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=True, high_resolution=False, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=False, high_resolution=False, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)

    from segment_clot_cta.prepare_training_dataset.get_sequence_pe_v2 import form_sample_sequence_dataset

    top_dict_dataset = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/'
    sample_sequence_save_dict = top_dict_dataset + 'sample_sequence/pe_v2/'
    form_sample_sequence_dataset(denoise=True, high_resolution=True, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=True, high_resolution=False, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=False, high_resolution=True, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
    form_sample_sequence_dataset(denoise=False, high_resolution=False, fold=current_fold,
                                 save_dict=sample_sequence_save_dict, dataset_dict=top_dict_dataset)
