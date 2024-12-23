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
import numpy as np
import os


def load_func_rsna(path_file):
    rescaled_cta = np.load(path_file)['array']
    print("clipping data to [-1000 HU, 100 HU]")
    clipped_cta = np.clip(rescaled_cta, -0.25, Functions.change_to_rescaled(100))
    return clipped_cta


def pipeline_process_all(top_dict_database, fold=(0, 1), batch_size=2, artery_vein=True, blood_high_recall=False,
                         denoise_ct=False):

    # experiments did not find evidence supporting that
    # using denoise or vessel high recall will improved the performance of PE analysis

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
    else:
        top_dict_rescaled_ct_denoise = top_dict_rescaled_ct

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=artery_vein,
                                                    load_func=load_func_rsna)

    if artery_vein or blood_high_recall:
        add_high_recall_blood_segmentation.process_dataset(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                           top_dict_semantics, fold=fold)

    # get blood depth array
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=blood_high_recall,
                                    process_av=False)

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, blood_high_recall=blood_high_recall,
                                     process_av=False)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=blood_high_recall,
                                    process_av=False)
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


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    Functions.set_visible_device('1')
    pipeline_process_all('/data_disk/RSNA-PE_dataset',
                         fold=(0, 8), artery_vein=True, blood_high_recall=False)
