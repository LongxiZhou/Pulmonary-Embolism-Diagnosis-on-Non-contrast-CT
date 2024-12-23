import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')

import chest_ct_database.feature_manager.add_basic_tissue_seg as basic_tissue_seg
import chest_ct_database.feature_manager.add_depth_array as add_depth_array
import chest_ct_database.feature_manager.add_center_lines as add_center_lines
import chest_ct_database.feature_manager.add_denoise_ct as add_denoise
import chest_ct_database.feature_manager.add_branch_map as add_branch_map
import chest_ct_database.visualize_manager.add_basic_tissue_visualize as add_basic_tissue_visualize
import chest_ct_database.visualize_manager.add_basic_tissue_visualize_3_view as add_basic_tissue_visualize_3_view
import chest_ct_database.report_manager.establish_report_dict as establish_report_dict
import chest_ct_database.report_manager.update_report_dict as update_report_dict
import chest_ct_database.pipeline_and_synchronize.pipeline_secondary_features as pipeline_secondary_features
import os
import chest_ct_database.feature_manager.add_high_recall_blood_segmenation as add_high_recall_blood_segmentation
from pe_dataset_management.basic_functions import get_dataset_relative_path
from functools import partial


def pipeline_process_all(top_dict_database, fold=(0, 1), batch_size=None, artery_vein=True,
                         for_twice_register_only=False):
    if batch_size is None:
        import torch
        batch_size = torch.cuda.device_count()
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_rescaled_ct = os.path.join(top_dict_database, 'rescaled_ct')
    if os.path.exists(top_dict_rescaled_ct):
        # denoise the CT data
        add_denoise.add_denoise_ct(top_dict_rescaled_ct, top_dict_rescaled_ct_denoise, fold=fold, batch_size=batch_size)

    if not os.path.exists(top_dict_rescaled_ct_denoise):
        print("no directory:", top_dict_rescaled_ct_denoise)
        return None

    if len(os.listdir(top_dict_rescaled_ct_denoise)) == 0:
        print("no files in directory:", top_dict_rescaled_ct_denoise)
        return None

    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')
    top_dict_reports = os.path.join(top_dict_database, 'reports')
    top_dict_visualize = os.path.join(top_dict_database, 'visualization/basic_semantic_check')
    top_dict_visualize_3_view = os.path.join(top_dict_database, 'visualization/basic_semantic_check_3_view')

    # segment lung, airway, blood vessel, artery, vein, nodules, infection, heart
    basic_tissue_seg.segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_semantics, fold=fold,
                                                    batch_size=batch_size, artery_vein=artery_vein)

    if artery_vein:
        add_high_recall_blood_segmentation.process_dataset(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                           top_dict_semantics, fold=fold)

    # get blood depth array
    print(top_dict_center_line_and_depth)
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=False,
                                    process_av=artery_vein)
    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=True,
                                    process_av=artery_vein)

    if for_twice_register_only:
        return None

    # get blood and airway center line mask
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, blood_high_recall=False,
                                     process_av=artery_vein)
    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, blood_high_recall=True,
                                     process_av=artery_vein)

    # get report for semantics
    if fold[0] == 0:
        if os.path.exists(top_dict_rescaled_ct_denoise):
            establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)
            update_report_dict.update_inherent_noise_database(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                              top_dict_reports, denoise=True)
        establish_report_dict.establish_report_dict_database(top_dict_rescaled_ct_denoise, denoise=True)

    pipeline_secondary_features.pipeline_process_all(top_dict_database, fold, process_av=artery_vein)

    # get blood branch map
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=False,
                                    process_av=artery_vein)
    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=True,
                                    process_av=artery_vein)
    # add basic tissue visualize
    add_basic_tissue_visualize.add_visualization_basic_semantic(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                                top_dict_visualize, fold=fold)

    # add basic tissue visualize
    add_basic_tissue_visualize_3_view.add_visualization_basic_semantic(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                                       top_dict_visualize_3_view, fold=fold)


def blood_high_recall_secondary(top_dict_database, fold=(0, 1)):
    pipeline_secondary_features.pipeline_process_all(top_dict_database, fold, process_av=False, blood_high_recall=True)


def process_blood_high_recall(top_dict_database, fold=(0, 1)):
    top_dict_rescaled_ct_denoise = os.path.join(top_dict_database, 'rescaled_ct-denoise')
    top_dict_semantics = os.path.join(top_dict_database, 'semantics')
    top_dict_center_line_and_depth = os.path.join(top_dict_database, 'depth_and_center-line')

    add_high_recall_blood_segmentation.process_dataset(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                                       top_dict_semantics, fold=fold)

    add_depth_array.add_depth_array(top_dict_rescaled_ct_denoise, top_dict_semantics,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=True,
                                    process_av=False)

    add_center_lines.add_center_line(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_center_line_and_depth,
                                     fold=fold, blood_high_recall=True,
                                     process_av=False)

    add_branch_map.add_branch_array(top_dict_rescaled_ct_denoise, top_dict_center_line_and_depth,
                                    top_dict_center_line_and_depth, fold=fold, blood_high_recall=True,
                                    process_av=False)


def semantic_for_all(fold, operation_func=None, artery_vein=True,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=True):
    """

    :param simulated_non_contrast:
    :param top_dict_database:
    :param fold:
    :param operation_func:
    :param artery_vein:

    :return:
    """

    if operation_func is None:
        operation_func = partial(pipeline_process_all, artery_vein=artery_vein)

    name_list_dataset = get_dataset_relative_path()
    for dataset in name_list_dataset:
        if simulated_non_contrast:
            top_dict = os.path.join(top_dict_database, dataset, 'simulated_non_contrast')
        else:
            top_dict = os.path.join(top_dict_database, dataset)
        operation_func(top_dict, fold=fold)


def prepare_semantic_for_twice_register(fold, top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA'):
    """

    :param top_dict_database:
    :param fold:

    :return:
    """

    operation_func = partial(pipeline_process_all, artery_vein=True, for_twice_register_only=True)

    name_list_dataset = get_dataset_relative_path()
    for dataset in name_list_dataset:
        top_dict = os.path.join(top_dict_database, dataset, 'smooth_register')
        operation_func(top_dict, fold=fold)


if __name__ == '__main__':
    # treat them as separate databases
    current_fold = (0, 1)
    visible_devices = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    process_av = True

    """
    # stage one, segment tissue for non-contrast and CTA
    """

    semantic_for_all(fold=current_fold, artery_vein=process_av,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=False)

    semantic_for_all(fold=current_fold, artery_vein=process_av,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_non_contrast',
                     simulated_non_contrast=False)

    """
    # stage two, get simulated non-contrast for CTA
    """

    semantic_for_all(fold=current_fold, artery_vein=process_av,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=True)

    """
    # stage three, 
    """

    semantic_for_all(fold=current_fold, artery_vein=process_av, operation_func=blood_high_recall_secondary,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=False)

    semantic_for_all(fold=current_fold, artery_vein=process_av, operation_func=blood_high_recall_secondary,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_non_contrast',
                     simulated_non_contrast=False)

    semantic_for_all(fold=current_fold, artery_vein=process_av, operation_func=blood_high_recall_secondary,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=True)
    exit()

    from smooth_mask.get_lung_vessel_blood_region.extract_training_sample_v2 import get_stack_array_256_version2

    get_stack_array_256_version2(fold=current_fold)

    semantic_for_all(fold=current_fold, artery_vein=True,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', simulated_non_contrast=False)
    semantic_for_all(fold=current_fold, artery_vein=True,
                     top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_non_contrast',
                     simulated_non_contrast=False)
