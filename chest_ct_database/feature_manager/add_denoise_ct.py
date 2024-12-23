"""
from feature rescaled_ct add denoise ct
"""

import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import chest_ct_database.public_datasets.RAD_ChestCT_dataset as rad_dataset
import collaborators_package.denoise_chest_ct.denoise_predict as de_noising
from functools import partial
import numpy as np


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, model=None, batch_size=2):
    """

    get denoise rescaled ct

    :param list_top_dict_reference: [top_dict_for_file_names]
    :param dataset_sub_dir:
    :param file_name:
    :param model: the loaded denoise model_or_model_path
    :param batch_size
    :return: denoise rescaled ct, in [512, 512, 512]
    """

    assert model is not None  # you have to pass a loaded model_or_model_path

    file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    print("loading:", file_path)

    if file_path[-1] == 'z':
        assert file_path[:-4] + '.npz' == file_path
        rescaled_ct = np.load(file_path)['array']
    else:
        assert file_path[:-4] + '.npy' == file_path
        rescaled_ct = np.load(file_path)

    denoise_rescaled_ct = de_noising.denoise_rescaled_array(rescaled_ct, model, batch_size=batch_size)

    return denoise_rescaled_ct


def func_file_operation_rad(list_top_dict_reference, dataset_sub_dir, file_name, model=None, batch_size=2):
    """

    get denoise rescaled ct from RAD-ChestCT dataset

    :param list_top_dict_reference: [top_dict_for_file_names]
    :param dataset_sub_dir:
    :param file_name:
    :param model: the loaded denoise model_or_model_path
    :param batch_size
    :return: denoise rescaled ct, in [512, 512, 512]
    """

    assert model is not None  # you have to pass a loaded model_or_model_path

    file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    print("loading:", file_path)

    assert file_path[:-4] + '.npz' == file_path
    rescaled_ct = rad_dataset.load_func_for_ct(file_path)

    denoise_rescaled_ct = de_noising.denoise_rescaled_array(rescaled_ct, model, batch_size=batch_size)

    return denoise_rescaled_ct


def func_file_save(save_dict, file_name, feature_package):
    Functions.save_np_array(save_dict, file_name[:-4] + '.npz', feature_package, compress=True, dtype='float16')


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def add_denoise_ct(top_dict_rescaled_ct, top_dict_save, fold=(0, 1), batch_size=None):
    if batch_size is None:
        import torch
        batch_size = torch.cuda.device_count()
    denoise_model = de_noising.load_model()
    func_file_operation_semantic = partial(func_file_operation, model=denoise_model,
                                           batch_size=batch_size)
    add_features.func_add_feature(top_dict_rescaled_ct, [top_dict_rescaled_ct], top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


def add_denoise_ct_rad(top_dict_stack_ct_rad_format, top_dict_save, fold=(0, 1)):
    denoise_model = de_noising.load_model()
    func_file_operation_semantic = partial(func_file_operation_rad, model=denoise_model, batch_size=2)
    add_features.func_add_feature(top_dict_stack_ct_rad_format, [top_dict_stack_ct_rad_format], top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    current_fold = (0, 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)
    add_denoise_ct('/data_disk/RSNA-PE_dataset/rescaled_ct/',
                   '/data_disk/RSNA-PE_dataset/rescaled_ct_denoise/', fold=current_fold)
