import pulmonary_embolism.prepare_dataset.get_branch_mask as get_branch_mask
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import numpy as np
from functools import partial


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, process_av=False, only_process_av=False,
                        blood_high_recall=False):
    """

    get branch_array for the given semantic

    :param blood_high_recall: if True, use the blood_mask_high_recall,
                                mostly used by CTA, as CTA has very high variance.
    :param only_process_av:
    :param process_av:
    :param list_top_dict_reference: [top_dict_depth_and_center-line]
    :param dataset_sub_dir:
    :param file_name:
    :return: semantic center line mask
    """

    def get_branch_map_artery_and_vein():
        file_path_artery_center_line = os.path.join(
            list_top_dict_reference[0], dataset_sub_dir, "artery_center_line", file_name)
        file_path_artery_depth_array = os.path.join(
            list_top_dict_reference[0], dataset_sub_dir, "depth_array_artery", file_name)
        print("loading center line:", file_path_artery_center_line)
        print("loading depth array:", file_path_artery_depth_array)
        depth_array_artery = np.load(file_path_artery_depth_array)['array']
        artery_center_line = np.load(file_path_artery_center_line)['array']
        branch_map_artery = get_branch_mask.get_branching_cloud(artery_center_line, depth_array_artery, search_radius=5,
                                                                smooth_radius=1,
                                                                step=1, weight_half_decay=20, refine_radius=4)
        file_path_vein_center_line = os.path.join(
            list_top_dict_reference[0], dataset_sub_dir, "vein_center_line", file_name)
        file_path_vein_depth_array = os.path.join(
            list_top_dict_reference[0], dataset_sub_dir, "depth_array_vein", file_name)
        print("loading center line:", file_path_vein_center_line)
        print("loading depth array:", file_path_vein_depth_array)
        depth_array_vein = np.load(file_path_vein_depth_array)['array']
        vein_center_line = np.load(file_path_vein_center_line)['array']
        branch_map_vein = get_branch_mask.get_branching_cloud(vein_center_line, depth_array_vein, search_radius=5,
                                                              smooth_radius=1,
                                                              step=1, weight_half_decay=20, refine_radius=4)
        return branch_map_artery, branch_map_vein

    def get_branch_map_blood():
        if blood_high_recall:
            file_path_blood_center_line = os.path.join(
                list_top_dict_reference[0], dataset_sub_dir, "blood_high_recall_center_line", file_name)
            file_path_blood_depth_array = os.path.join(
                list_top_dict_reference[0], dataset_sub_dir, "high_recall_depth_array", file_name)
        else:
            file_path_blood_center_line = os.path.join(
                list_top_dict_reference[0], dataset_sub_dir, "blood_center_line", file_name)
            file_path_blood_depth_array = os.path.join(
                list_top_dict_reference[0], dataset_sub_dir, "depth_array", file_name)

        print("loading center line:", file_path_blood_center_line)
        print("loading depth array:", file_path_blood_depth_array)

        depth_array = np.load(file_path_blood_depth_array)['array']
        blood_center_line = np.load(file_path_blood_center_line)['array']

        branch_map = get_branch_mask.get_branching_cloud(blood_center_line, depth_array, search_radius=5,
                                                         smooth_radius=1,
                                                         step=1, weight_half_decay=20, refine_radius=4)
        return branch_map

    if only_process_av:
        return get_branch_map_artery_and_vein()

    if process_av:
        artery_branch, vein_branch = get_branch_map_artery_and_vein()
        blood_branch = get_branch_map_blood()
        return artery_branch, vein_branch, blood_branch

    blood_branch = get_branch_map_blood()
    return blood_branch


def func_file_save(save_dict, file_name, feature_package, process_av=False, only_process_av=False,
                   blood_high_recall=False):
    if only_process_av:
        save_dict_0 = os.path.join(save_dict, "artery_branch_map")
        Functions.save_np_array(save_dict_0, file_name[:-4] + '.npz', feature_package[0], compress=True)
        save_dict_1 = os.path.join(save_dict, "vein_branch_map")
        Functions.save_np_array(save_dict_1, file_name[:-4] + '.npz', feature_package[1], compress=True)
        return None
    if process_av:
        if blood_high_recall:
            save_dict_2 = os.path.join(save_dict, "high_recall_blood_branch_map")
        else:
            save_dict_2 = os.path.join(save_dict, "blood_branch_map")
        Functions.save_np_array(save_dict_2, file_name[:-4] + '.npz', feature_package[2], compress=True)
        save_dict_0 = os.path.join(save_dict, "artery_branch_map")
        Functions.save_np_array(save_dict_0, file_name[:-4] + '.npz', feature_package[0], compress=True)
        save_dict_1 = os.path.join(save_dict, "vein_branch_map")
        Functions.save_np_array(save_dict_1, file_name[:-4] + '.npz', feature_package[1], compress=True)
        return None
    if blood_high_recall:
        save_dict = os.path.join(save_dict, "high_recall_blood_branch_map")
    else:
        save_dict = os.path.join(save_dict, "blood_branch_map")
    Functions.save_np_array(save_dict, file_name[:-4] + '.npz', feature_package, compress=True)


def func_check_processed(save_dict, file_name, process_av=False, blood_high_recall=False):
    if process_av:
        save_dict = os.path.join(save_dict, "vein_branch_map")
        path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
        if not os.path.exists(path_saved):
            return False
        return func_check_processed(Functions.get_father_dict(save_dict), file_name, False, blood_high_recall)
    if blood_high_recall:
        save_dict = os.path.join(save_dict, "high_recall_blood_branch_map")
    else:
        save_dict = os.path.join(save_dict, "blood_branch_map")
    path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def add_branch_array(top_dict_source, top_dict_semantics, top_dict_save, fold=(0, 1),
                     process_av=False, only_process_av=False, blood_high_recall=False):
    if only_process_av:
        assert not blood_high_recall
        func_file_op = partial(func_file_operation, only_process_av=True, process_av=True,
                               blood_high_recall=False)
        func_file_sa = partial(func_file_save, only_process_av=True, process_av=True,
                               blood_high_recall=False)
        func_check_process = partial(func_check_processed, process_av=True, blood_high_recall=False)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    if not process_av:
        func_file_op = partial(func_file_operation, only_process_av=False, process_av=False,
                               blood_high_recall=blood_high_recall)
        func_file_sa = partial(func_file_save, only_process_av=False, process_av=False,
                               blood_high_recall=blood_high_recall)
        func_check_process = partial(func_check_processed, process_av=False,
                                     blood_high_recall=blood_high_recall)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    func_file_op = partial(func_file_operation, only_process_av=False, process_av=True,
                           blood_high_recall=blood_high_recall)
    func_file_sa = partial(func_file_save, only_process_av=False, process_av=True,
                           blood_high_recall=blood_high_recall)
    func_check_process = partial(func_check_processed, process_av=True,
                                 blood_high_recall=blood_high_recall)

    add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                  func_file_sa, func_check_processed=func_check_process, fold=fold)
    return None


if __name__ == '__main__':
    current_fold = (0, 30)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)
    add_branch_array('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                     '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/',
                     '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/',
                     fold=current_fold, process_av=False, blood_high_recall=False)
    add_branch_array('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                     '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/',
                     '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/',
                     fold=current_fold, process_av=False, blood_high_recall=True)
    exit()
    add_branch_array('/data_disk/artery_vein_project/new_data/non-contrast/rescaled_ct-denoise/',
                     '/data_disk/artery_vein_project/new_data/non-contrast/depth_and_center-line/',
                     '/data_disk/artery_vein_project/new_data/non-contrast/depth_and_center-line/',
                     fold=(0, 10), only_process_av=True)
    exit()
    add_branch_array('/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/denoise-rescaled_ct/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/depth_and_center-line/',
                     '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/depth_and_center-line/',
                     fold=(0, 5))
