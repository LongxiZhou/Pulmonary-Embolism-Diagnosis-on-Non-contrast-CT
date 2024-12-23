import smooth_mask.get_lung_vessel_blood_region.inference as get_blood_region
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
from functools import partial
import numpy as np

"""
apply the following to existing blood region:

# blood region for non-contrast should in [-150, 150], here rescaled_ct has to be de-noised
valid_region = np.array(rescaled_ct > Functions.change_to_rescaled(-150), 'float32')
valid_region = valid_region * np.array(rescaled_ct < Functions.change_to_rescaled(150), 'float32')

# trim the big vessel
loc_list_blood_region_rescaled = Functions.get_location_list(np.where(blood_region_rescaled > 0.5))
for loc in loc_list_blood_region_rescaled:
    if depth_of_raw_vessel_mask[loc] < (7 - blood_branch_map[loc]):
        blood_region_rescaled[loc] = 0
blood_region_rescaled = blood_region_rescaled * valid_region

# smooth the mask
blood_region_rescaled = blood_region_rescaled + get_surface(blood_region_rescaled, outer=True, strict=True)
blood_region_rescaled = blood_region_rescaled - get_surface(blood_region_rescaled, outer=False, strict=True)
"""


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, process_av=False, only_process_av=False,
                        visible_device='0', blood_high_recall=False):
    """

    get blood region for the given semantic

    :param blood_high_recall: blood vessel is from blood_mask_high_recall
    :param visible_device:
    :param list_top_dict_reference: [top_dict_semantic, top_dict_rescaled_ct_denoise, top_dict_secondary_semantics,
                                    top_dict_secondary_semantics]
    :param dataset_sub_dir:
    :param file_name:
    :param process_av:
    :param only_process_av: higher priority than "process_av"
    :return: center_line_mask_airway, center_line_mask_blood
    """

    def get_blood_region_semantic(semantic):
        if blood_high_recall:
            if semantic == "blood_mask":
                semantic = "blood_mask_high_recall"
        file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, semantic, file_name)
        rescaled_ct_path = os.path.join(list_top_dict_reference[1], dataset_sub_dir, file_name)
        if blood_high_recall:
            blood_branch_map_path = os.path.join(
                list_top_dict_reference[2], dataset_sub_dir, 'high_recall_blood_branch_map', file_name)
            vessel_depth_path = os.path.join(
                list_top_dict_reference[3], dataset_sub_dir, 'high_recall_depth_array', file_name)
        else:
            blood_branch_map_path = os.path.join(
                list_top_dict_reference[2], dataset_sub_dir, 'blood_branch_map', file_name)
            vessel_depth_path = os.path.join(list_top_dict_reference[3], dataset_sub_dir, 'depth_array', file_name)
        print("loading:", file_path)
        vessel_mask = np.load(file_path)['array']
        rescaled_ct = np.load(rescaled_ct_path)['array']
        blood_branch_map = np.load(blood_branch_map_path)['array']
        vessel_depth = np.load(vessel_depth_path)['array']
        print("get", semantic, "blood region")
        blood_region_mask = get_blood_region.get_blood_region_rescaled_mask(
            vessel_mask, rescaled_ct, get_connect=True, visible_device=visible_device,
            refine_reference=(vessel_depth, blood_branch_map), get_connect_final=False)
        return blood_region_mask

    if only_process_av:
        return get_blood_region_semantic("artery_mask"), get_blood_region_semantic("vein_mask")

    if not process_av:
        return get_blood_region_semantic("blood_mask"),

    return get_blood_region_semantic("artery_mask"), get_blood_region_semantic("vein_mask"), get_blood_region_semantic(
        "blood_mask")


def func_file_save(save_dict, file_name, feature_package, process_av=False, only_process_av=False,
                   blood_high_recall=False):
    def save_av():
        save_dict_artery = os.path.join(save_dict, 'artery_blood_region_strict')
        Functions.save_np_array(save_dict_artery, file_name[:-4] + '.npz', feature_package[0], compress=True)

        save_dict_vein = os.path.join(save_dict, 'vein_blood_region_strict')
        Functions.save_np_array(save_dict_vein, file_name[:-4] + '.npz', feature_package[1], compress=True)

    def save_blood(shift=0):
        if blood_high_recall is False:
            save_dict_blood = os.path.join(save_dict, 'blood_region_strict')
        else:
            save_dict_blood = os.path.join(save_dict, 'blood_region_strict_high_recall')
        Functions.save_np_array(save_dict_blood, file_name[:-4] + '.npz', feature_package[shift], compress=True)

    if only_process_av:
        save_av()
        return None

    if not process_av:
        save_blood(0)
        return None

    save_blood(2)
    save_av()


def func_check_processed(save_dict, file_name, process_av=False, blood_high_recall=False):
    if not process_av:
        if not blood_high_recall:
            save_dict_blood = os.path.join(save_dict, 'blood_region_strict')
        else:
            save_dict_blood = os.path.join(save_dict, 'blood_region_strict_high_recall')
        path_saved = os.path.join(save_dict_blood, file_name[:-4] + '.npz')
        if os.path.exists(path_saved):
            return True
        return False
    save_dict_vein = os.path.join(save_dict, 'vein_blood_region_strict')
    path_saved = os.path.join(save_dict_vein, file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def add_blood_region_strict(top_dict_rescaled_ct, top_dict_semantics, top_dict_save, top_dict_depth, fold=(0, 1),
                            process_av=False, only_process_av=False, visible_device=None, blood_high_recall=False):

    reference_list = [top_dict_semantics, top_dict_rescaled_ct, top_dict_save, top_dict_depth]

    if only_process_av:
        func_file_op = partial(func_file_operation, only_process_av=True, process_av=True,
                               visible_device=visible_device, blood_high_recall=blood_high_recall)
        func_file_sa = partial(func_file_save, only_process_av=True, process_av=True,
                               blood_high_recall=blood_high_recall)
        func_check_process = partial(func_check_processed, process_av=True, blood_high_recall=blood_high_recall)

        add_features.func_add_feature(top_dict_rescaled_ct, reference_list, top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    if not process_av:
        func_file_op = partial(func_file_operation, only_process_av=False, process_av=False,
                               visible_device=visible_device, blood_high_recall=blood_high_recall)
        func_file_sa = partial(func_file_save, only_process_av=False, process_av=False,
                               blood_high_recall=blood_high_recall)
        func_check_process = partial(func_check_processed, process_av=False,
                                     blood_high_recall=blood_high_recall)

        add_features.func_add_feature(top_dict_rescaled_ct, reference_list, top_dict_save, func_file_op,
                                      func_file_sa, func_check_processed=func_check_process, fold=fold)
        return None

    func_file_op = partial(func_file_operation, only_process_av=False, process_av=True,
                           visible_device=visible_device, blood_high_recall=blood_high_recall)
    func_file_sa = partial(func_file_save, only_process_av=False, process_av=True,
                           blood_high_recall=blood_high_recall)
    func_check_process = partial(func_check_processed, process_av=True, blood_high_recall=blood_high_recall)

    add_features.func_add_feature(top_dict_rescaled_ct, reference_list, top_dict_save, func_file_op,
                                  func_file_sa, func_check_processed=func_check_process, fold=fold)
    return None


if __name__ == '__main__':
    add_blood_region_strict('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise/',
                            '/data_disk/RAD-ChestCT_dataset/semantics/',
                            '/data_disk/RAD-ChestCT_dataset/secondary_semantics/',
                            '/data_disk/RAD-ChestCT_dataset/secondary_semantics/',
                            fold=(1, 2),
                            process_av=False, visible_device='0')
