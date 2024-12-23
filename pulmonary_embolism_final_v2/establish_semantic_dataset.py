import chest_ct_database.feature_manager.add_basic_tissue_seg as basic_tissue_seg
import chest_ct_database.feature_manager.add_denoise_ct as add_denoise
import chest_ct_database.feature_manager.add_depth_array as add_depth_array
import chest_ct_database.feature_manager.add_center_lines as add_center_lines
import chest_ct_database.feature_manager.add_branch_map as add_branch_map
import chest_ct_database.visualize_manager.add_basic_tissue_visualize as add_basic_tissue_visualize
import chest_ct_database.report_manager.establish_report_dict as establish_report_dict
import chest_ct_database.report_manager.update_report_dict as update_report_dict
import os


def pipeline_process_all(top_dict_database, fold=(0, 1), batch_size=2):

    top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    top_dict_reports = os.path.join(top_dict_database, 'reports')
    top_dict_visualize = os.path.join(top_dict_database, 'visualization/basic_semantic_check')

    # denoise the CT data
    if os.path.exists(top_dict_rescaled_ct):
        add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold)

    assert os.path.exists(top_dict_rescaled_ct_denoise)

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=True)

    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold)

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold)

    # get report for semantics
    if fold[0] == 0:
        if os.path.exists(top_dict_rescaled_ct):
            establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct, denoise=False)
            update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                              top_dict_reports, denoise=True)
        establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold)
    # add basic tissue visualize
    add_basic_tissue_visualize.add_visualization_basic_semantic(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                                top_dict_visualize, fold=fold)


def customized(top_dict_database, fold=(0, 1), batch_size=2):
    top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    # top_dict_reports = os.path.join(top_dict_database, 'reports')

    # denoise the CT data
    add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold)

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, process_av=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, process_av=False)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, process_av=False)

    # get report for semantics
    # establish_report_dict.establish_report_dict_database(top_dict_dataset_non_contrast, denoise=False)
    # update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct_denoise, top_dict_semantics,
    #                                                  top_dict_reports, denoise=True)


if __name__ == '__main__':
    # treat them as separate databases
    current_fold = (0, 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality', fold=current_fold)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality', fold=current_fold)

    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                         'good_CTA-CT_interval_but_bad_dcm', fold=current_fold)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                         'CT-after-CTA', fold=current_fold)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                         'long_CTA-CT_interval', fold=current_fold)

    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                         'CT-after-CTA', fold=current_fold)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                         'long_CTA-CT_interval', fold=current_fold)
    pipeline_process_all('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                         'good_CTA-CT_interval_but_bad_dcm', fold=current_fold)
