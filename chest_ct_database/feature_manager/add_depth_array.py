import analysis.center_line_and_depth_3D as get_depth
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
from functools import partial
import numpy as np


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, process_av=False, only_process_av=False,
                        secondary=False, blood_high_recall=False):
    """

    get depth for the given semantic

    :param blood_high_recall: if True, use the blood_mask_high_recall,
                                mostly used by CTA, as CTA has very high variance.
    :param secondary:
    :param only_process_av:
    :param process_av:
    :param list_top_dict_reference: [top_dict_semantic]
    :param dataset_sub_dir:
    :param file_name:
    :return: semantic center line mask
    """

    def get_depth_semantic(semantic):
        if blood_high_recall:
            if semantic == 'blood_mask':
                semantic = 'blood_mask_high_recall'
            if semantic == 'blood_region':
                semantic = 'blood_region_high_recall'
        file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, semantic, file_name)
        print("loading:", file_path)
        vessel_mask = np.load(file_path)['array']
        depth_array = get_depth.get_surface_distance(vessel_mask)
        depth_array = np.array(depth_array, 'float16')
        return depth_array

    if only_process_av:
        if secondary:
            return get_depth_semantic('artery_blood_region'), get_depth_semantic('vein_blood_region')
        return get_depth_semantic('artery_mask'), get_depth_semantic('vein_mask')

    if process_av:
        if secondary:
            return get_depth_semantic('artery_blood_region'), get_depth_semantic(
                'vein_blood_region'), get_depth_semantic('blood_region')
        return get_depth_semantic('artery_mask'), get_depth_semantic('vein_mask'), get_depth_semantic('blood_mask')
    if secondary:
        return get_depth_semantic('blood_region')
    return get_depth_semantic('blood_mask')


def func_file_save(save_dict, file_name, feature_package, process_av=False, only_process_av=False,
                   blood_high_recall=False):
    if only_process_av:
        save_dict_0 = os.path.join(save_dict, "depth_array_artery")
        Functions.save_np_array(save_dict_0, file_name[:-4] + '.npz', feature_package[0], compress=True)
        save_dict_1 = os.path.join(save_dict, "depth_array_vein")
        Functions.save_np_array(save_dict_1, file_name[:-4] + '.npz', feature_package[1], compress=True)
        return None
    if process_av:
        if blood_high_recall:
            save_dict_2 = os.path.join(save_dict, "high_recall_depth_array")
        else:
            save_dict_2 = os.path.join(save_dict, "depth_array")
        Functions.save_np_array(save_dict_2, file_name[:-4] + '.npz', feature_package[2], compress=True)
        save_dict_0 = os.path.join(save_dict, "depth_array_artery")
        Functions.save_np_array(save_dict_0, file_name[:-4] + '.npz', feature_package[0], compress=True)
        save_dict_1 = os.path.join(save_dict, "depth_array_vein")
        Functions.save_np_array(save_dict_1, file_name[:-4] + '.npz', feature_package[1], compress=True)
        return None
    if blood_high_recall:
        save_dict = os.path.join(save_dict, "high_recall_depth_array")
    else:
        save_dict = os.path.join(save_dict, "depth_array")
    Functions.save_np_array(save_dict, file_name[:-4] + '.npz', feature_package, compress=True)


def func_check_processed(save_dict, file_name, process_av=False, blood_high_recall=False):
    if process_av:
        save_dict = os.path.join(save_dict, "depth_array_vein")
        path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
        if not os.path.exists(path_saved):
            print("path", path_saved, "not exist")
            return False
        return func_check_processed(Functions.get_father_dict(save_dict), file_name, False, blood_high_recall)
    if blood_high_recall:
        save_dict = os.path.join(save_dict, "high_recall_depth_array")
    else:
        save_dict = os.path.join(save_dict, "depth_array")
    path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    print("path", path_saved, "not exist")
    return False


def add_depth_array(top_dict_source, top_dict_semantics, top_dict_save, fold=(0, 1),
                    process_av=False, only_process_av=False, secondary=False, blood_high_recall=False):
    if only_process_av:
        assert blood_high_recall is False
        func_file_op = partial(func_file_operation, only_process_av=True, process_av=True, secondary=secondary,
                               blood_high_recall=False)
        func_file_sa = partial(func_file_save, only_process_av=True, process_av=True,
                               blood_high_recall=False)
        func_check_process = partial(func_check_processed, process_av=True, blood_high_recall=False)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    if not process_av:
        func_file_op = partial(func_file_operation, only_process_av=False, process_av=False, secondary=secondary,
                               blood_high_recall=blood_high_recall)
        func_file_sa = partial(func_file_save, only_process_av=False, process_av=False,
                               blood_high_recall=blood_high_recall)
        func_check_process = partial(func_check_processed, process_av=False,
                                     blood_high_recall=blood_high_recall)

        add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    func_file_op = partial(func_file_operation, only_process_av=False, process_av=True, secondary=secondary,
                           blood_high_recall=blood_high_recall)
    func_file_sa = partial(func_file_save, only_process_av=False, process_av=True,
                           blood_high_recall=blood_high_recall)
    func_check_process = partial(func_check_processed, process_av=True,
                                 blood_high_recall=blood_high_recall)

    add_features.func_add_feature(top_dict_source, [top_dict_semantics], top_dict_save, func_file_op,
                                  func_file_sa, func_check_processed=func_check_process, fold=fold)
    return None


if __name__ == '__main__':
    current_fold = (0, 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    add_depth_array('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/',
                    '/data_disk/RSNA-PE_dataset/simulated_non_contrast/depth_and_center-line/', fold=current_fold,
                    process_av=False, secondary=False, blood_high_recall=True)

    exit()
    add_depth_array('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise/',
                    '/data_disk/RAD-ChestCT_dataset/secondary_semantics/',
                    '/data_disk/RAD-ChestCT_dataset/secondary_semantics/', fold=(0, 2),
                    process_av=False, secondary=True, blood_high_recall=False)
