import numpy as np
import Tool_Functions.Functions as Functions
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import os


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name):
    file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    print("loading:", file_path)
    if file_path[-1] == 'y':
        rescaled_ct = np.load(file_path)
    else:
        rescaled_ct = np.load(file_path)['array']
    new_array = convert_rescaled_ct_to_float16(rescaled_ct)
    return new_array


def func_file_save(save_dict, file_name, feature_package):
    Functions.save_np_array(save_dict, file_name[:-4] + '.npz', feature_package, compress=True)


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def convert_rescaled_ct_to_float16(rescaled_ct):
    rescaled_ct = np.array(rescaled_ct * 1600, 'int16')
    new_array = np.array(rescaled_ct, 'float16') / 1600
    return new_array


def save_to_float16(top_dict_source, top_dict_save, fold=(0, 1)):
    add_features.func_add_feature(top_dict_source, [top_dict_source], top_dict_save, func_file_operation,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    save_to_float16('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct',
                    '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16', (0, 1))
