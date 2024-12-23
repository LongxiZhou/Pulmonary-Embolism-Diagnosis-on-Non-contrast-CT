from chest_ct_database.initialize_or_pipeline_process.pipeline_dcm_to_npz import dcm_folders_to_rescaled_array, \
    get_save_name_av_dataset
from chest_ct_database.initialize_or_pipeline_process.pipeline_mha_to_npz_and_check_gt import mha_file_to_rescaled_gt, \
    pipeline_check_av_an_dataset, whether_processed_av_an_dataset, save_func_av_an_dataset
from chest_ct_database.initialize_or_pipeline_process.pipeline_rescaled_ct_to_all import pipeline_process_all


# only for temp use. check the functions before call
if __name__ == '__main__':
    fold = (0, 4)

    dcm_folders_to_rescaled_array('/data_disk/artery_vein_project/new_data/CTA/dcm_files/',
                                  get_save_name_av_dataset,
                                  '/data_disk/artery_vein_project/new_data/CTA/rescaled_ct/',
                                  denoise=False,
                                  exclusion_func=None, fold=fold)

    mha_file_to_rescaled_gt('/data_disk/artery_vein_project/new_data/CTA/dcm_files',
                            get_save_name_av_dataset,
                            '/data_disk/artery_vein_project/new_data/CTA/ground_truth',
                            whether_processed_av_an_dataset,
                            save_func_av_an_dataset, exclusion_func=None, fold=fold)

    pipeline_check_av_an_dataset('/data_disk/artery_vein_project/new_data/CTA/rescaled_ct/',
                                 '/data_disk/artery_vein_project/new_data/CTA/ground_truth/',
                                 '/data_disk/artery_vein_project/new_data/CTA/visualization/check_gt/',
                                 fold=fold)

    pipeline_process_all('/data_disk/artery_vein_project/new_data/CTA', fold=fold)
