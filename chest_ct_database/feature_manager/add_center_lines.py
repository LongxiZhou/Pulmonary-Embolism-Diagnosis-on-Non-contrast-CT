import analysis.center_line_and_depth_3D as get_center_line
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
from functools import partial
import numpy as np


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, process_av=False, only_process_av=False,
                        secondary=False, blood_high_recall=False, complete_semantics=None):
    """

    get center_line for the given semantic

    :param complete_semantics:
    :param blood_high_recall: if True, use the blood_mask_high_recall,
                                mostly used by CTA, as CTA has very high variance.
    :param secondary:
    :param list_top_dict_reference: [top_dict_semantic]
    :param dataset_sub_dir:
    :param file_name:
    :param process_av:
    :param only_process_av: higher priority than "process_av"
    :return: center_line_mask_airway, center_line_mask_blood
    """
    if complete_semantics is None:
        complete_semantics = []
    else:
        complete_semantics = list(complete_semantics)

    def get_center_line_semantic(semantic):
        if semantic in complete_semantics:
            print(semantic, "completed")
            return None

        if blood_high_recall:
            if semantic == 'blood_mask':
                semantic = 'blood_mask_high_recall'
            if semantic == 'blood_region':
                semantic = 'blood_region_high_recall'

        file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, semantic, file_name)
        print("loading:", file_path)
        vessel_mask = np.load(file_path)['array']
        print("get", semantic, "center line")
        center_line_mask = get_center_line.get_center_line(vessel_mask)
        return center_line_mask

    if only_process_av:
        if secondary:
            return get_center_line_semantic("artery_blood_region"), get_center_line_semantic("vein_blood_region")
        return get_center_line_semantic("artery_mask"), get_center_line_semantic("vein_mask")

    if not process_av:
        if secondary:
            return None, get_center_line_semantic("blood_region")
        return get_center_line_semantic("airway_mask"), get_center_line_semantic("blood_mask")

    if secondary:
        return get_center_line_semantic("artery_blood_region"), get_center_line_semantic(
            "vein_blood_region"), None, get_center_line_semantic("blood_region")

    return get_center_line_semantic("artery_mask"), get_center_line_semantic("vein_mask"), get_center_line_semantic(
        "airway_mask"), get_center_line_semantic("blood_mask")


def func_file_save(save_dict, file_name, feature_package, process_av=False, only_process_av=False,
                   blood_high_recall=False):
    def save_av():
        save_dict_artery = os.path.join(save_dict, 'artery_center_line')
        Functions.save_np_array(save_dict_artery, file_name[:-4] + '.npz', feature_package[0], compress=True)

        save_dict_vein = os.path.join(save_dict, 'vein_center_line')
        Functions.save_np_array(save_dict_vein, file_name[:-4] + '.npz', feature_package[1], compress=True)

    def save_airway_and_blood(shift=0):
        save_dict_airway = os.path.join(save_dict, 'airway_center_line')
        if feature_package[shift] is not None:
            Functions.save_np_array(save_dict_airway, file_name[:-4] + '.npz', feature_package[shift], compress=True)

        if blood_high_recall:
            save_dict_blood = os.path.join(save_dict, 'blood_high_recall_center_line')
        else:
            save_dict_blood = os.path.join(save_dict, 'blood_center_line')
        Functions.save_np_array(save_dict_blood, file_name[:-4] + '.npz', feature_package[shift + 1], compress=True)

    if only_process_av:
        save_av()
        return None

    if not process_av:
        save_airway_and_blood(0)
        return None

    save_airway_and_blood(2)
    save_av()


def func_check_processed(save_dict, file_name, process_av=False, blood_high_recall=False):
    if not process_av:
        if blood_high_recall:
            save_dict_blood = os.path.join(save_dict, 'blood_high_recall_center_line')
        else:
            save_dict_blood = os.path.join(save_dict, 'blood_center_line')
        path_saved = os.path.join(save_dict_blood, file_name[:-4] + '.npz')
        if os.path.exists(path_saved):
            return True
        return False
    save_dict_vein = os.path.join(save_dict, 'vein_center_line')
    path_saved = os.path.join(save_dict_vein, file_name[:-4] + '.npz')
    if not os.path.exists(path_saved):
        return False
    return func_check_processed(save_dict, file_name, False, blood_high_recall)


def add_center_line(top_dict_source, top_dict_semantics, top_dict_save, fold=(0, 1),
                    process_av=False, only_process_av=False, secondary=False, blood_high_recall=False,
                    complete_semantics=None):
    if only_process_av:
        assert blood_high_recall is False  # blood_high_recall is not a-v seg
        func_file_op = partial(func_file_operation, only_process_av=True, process_av=True, secondary=secondary,
                               blood_high_recall=False, complete_semantics=complete_semantics)
        func_file_sa = partial(func_file_save, only_process_av=True, process_av=True,
                               blood_high_recall=False)
        func_check_process = partial(func_check_processed, process_av=True, blood_high_recall=False)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    if not process_av:
        func_file_op = partial(func_file_operation, only_process_av=False, process_av=False,  secondary=secondary,
                               blood_high_recall=blood_high_recall, complete_semantics=complete_semantics)
        func_file_sa = partial(func_file_save, only_process_av=False, process_av=False,
                               blood_high_recall=blood_high_recall)
        func_check_process = partial(func_check_processed, process_av=False,
                                     blood_high_recall=blood_high_recall)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    func_file_op = partial(func_file_operation, only_process_av=False, process_av=True, secondary=secondary,
                           blood_high_recall=blood_high_recall, complete_semantics=complete_semantics)
    func_file_sa = partial(func_file_save, only_process_av=False, process_av=True,
                           blood_high_recall=blood_high_recall)
    func_check_process = partial(func_check_processed, process_av=True,
                                 blood_high_recall=blood_high_recall)

    add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                  func_file_sa, func_check_processed=func_check_process, fold=fold)
    return None


if __name__ == '__main__':
    current_fold = (0, 8)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)
    add_center_line('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/', fold=current_fold,
                    process_av=False, secondary=False, blood_high_recall=False)

    add_center_line('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/', fold=current_fold,
                    process_av=False, secondary=False, blood_high_recall=True, complete_semantics=["airway_mask"])
    exit()
    add_center_line('/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/',
                    '/data_disk/rescaled_ct_and_semantics/secondary_semantics/',
                    '/data_disk/rescaled_ct_and_semantics/secondary_semantics/', fold=(0, 2),
                    process_av=False, secondary=True, blood_high_recall=False)
