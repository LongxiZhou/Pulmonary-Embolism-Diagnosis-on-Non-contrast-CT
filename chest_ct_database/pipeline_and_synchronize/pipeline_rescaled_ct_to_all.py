import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import chest_ct_database.feature_manager.add_basic_tissue_seg as basic_tissue_seg
import chest_ct_database.feature_manager.add_denoise_ct as add_denoise
import chest_ct_database.feature_manager.add_depth_array as add_depth_array
import chest_ct_database.feature_manager.add_center_lines as add_center_lines
import chest_ct_database.feature_manager.add_branch_map as add_branch_map
import chest_ct_database.visualize_manager.add_basic_tissue_visualize as add_basic_tissue_visualize
import chest_ct_database.report_manager.establish_report_dict as establish_report_dict
import chest_ct_database.feature_manager.add_high_recall_blood_segmenation as add_high_recall_blood_segmentation

import os


def pipeline_process_all(top_dict_database, fold=(0, 1), batch_size=2, artery_vein=False, blood_high_recall=False,
                         denoise_ct=True):
    # for database, file structure like: top_dict_database/rescaled_ct/dataset_name

    top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    top_dict_reports = os.path.join(top_dict_database, 'reports')
    top_dict_visualize = os.path.join(top_dict_database, 'visualization/basic_semantic_check')

    # denoise the CT data
    if not os.path.exists(top_dict_rescaled_ct_denoise) and denoise_ct:
        add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold)

    if denoise_ct:
        assert os.path.exists(top_dict_rescaled_ct_denoise)

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=artery_vein)

    if artery_vein or blood_high_recall:
        add_high_recall_blood_segmentation.process_dataset(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                           top_dict_semantics, fold=fold)

    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=blood_high_recall,
                                    process_av=artery_vein)

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, blood_high_recall=blood_high_recall,
                                     process_av=artery_vein)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=blood_high_recall,
                                    process_av=artery_vein)
    # add basic tissue visualize
    add_basic_tissue_visualize.add_visualization_basic_semantic(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                                top_dict_visualize, fold=fold)

    # get report for semantics
    if fold[0] == 0:
        # if os.path.exists(top_dict_rescaled_ct):
        #     establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)
        #     update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct, top_dict_semantics,
        #                                                       top_dict_reports, denoise=False)
        establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)


def get_artery_vein_analysis(top_dict_database, fold=(0, 1)):
    # top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    # top_dict_reports = os.path.join(top_dict_database, 'reports')

    """
    # denoise the CT data
    add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold)

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=False)
    """

    # the a-v seg model is 3D and only inference scan by scan, so multiple GPU is not useful
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0] % 2)

    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, process_av=True)

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, process_av=True)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, process_av=True)

    # get report for semantics
    exit()
    if fold[0] == 0:
        establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)
        # update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct_denoise, top_dict_semantics,
        #                                                  top_dict_reports, denoise=True)


def debug_use(top_dict_database, fold=(0, 1), batch_size=2, artery_vein=False):
    # for database, file structure like: top_dict_database/rescaled_ct/dataset_name

    if artery_vein:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0] % 2)

    import chest_ct_database.visualize_manager.add_basic_tissue_visualize_3_view as add_basic_tissue_visualize_3_view

    top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    top_dict_reports = os.path.join(top_dict_database, 'reports')
    top_dict_visualize = os.path.join(top_dict_database, 'visualization/basic_semantic_check_three_view')

    # denoise the CT data
    if os.path.exists(top_dict_rescaled_ct):
        add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold)

    assert os.path.exists(top_dict_rescaled_ct_denoise)

    # add basic tissue visualize
    add_basic_tissue_visualize_3_view.add_visualization_basic_semantic(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                                       top_dict_visualize, fold=fold)


def temp_use(top_dict_database, fold=(0, 1), batch_size=2, artery_vein=False):
    # for database, file structure like: top_dict_database/rescaled_ct/dataset_name

    if artery_vein:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0] % 2)

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
                                                    batch_size=batch_size, artery_vein=artery_vein)

    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold)

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold)

    # get report for semantics
    if fold[0] == 0:
        # if os.path.exists(top_dict_rescaled_ct):
        #     establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)
        #     update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct, top_dict_semantics,
        #                                                       top_dict_reports, denoise=False)
        establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions
    Functions.set_visible_device('1')
    pipeline_process_all('/data_disk/RSNA-PE_dataset',
                         fold=(0, 8), artery_vein=True, blood_high_recall=False)
