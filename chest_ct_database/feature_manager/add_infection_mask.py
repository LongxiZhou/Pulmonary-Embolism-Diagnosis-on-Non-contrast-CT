
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
from functools import partial
import numpy as np

import basic_tissue_prediction.three_way_prediction as three_way_prediction
import basic_tissue_prediction.connectivity_refine as connectivity_refine


def predict_infection_masks_rescaled_array(rescaled_array, batch_size=2,
                                           threshold=2., loaded_models=None, array_info=None):
    """
    :param array_info:
    :param loaded_models:
    :param threshold:
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param batch_size: the batch_size when prediction

    :return: infection mask for the rescaled array, binary numpy array in shape [512, 512, 512], 0 outer infection 1 inner infection.
    """

    print("predicting infection masks\n")
    infection_mask = three_way_prediction.three_way_predict_binary_class_faster(
        rescaled_array, None, threshold, batch_size, loaded_models, array_info)
    del rescaled_array
    return infection_mask


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, model=None, batch_size=2, array_info=None):
    """

    :param array_info:
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

    infection_mask = predict_infection_masks_rescaled_array(
        rescaled_ct, batch_size=batch_size, loaded_models=model, array_info=array_info)

    return infection_mask


def func_file_save(save_dict, file_name, feature_package):
    Functions.save_np_array(os.path.join(save_dict, 'infection'), file_name[:-4] + '.npz', feature_package,
                            compress=True, dtype='float16')


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, 'infection', file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def add_infection_seg(top_dict_rescaled_ct, top_dict_save, fold=(0, 1)):
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-5, -2, 0, 2, 5),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    check_point_dict = '/home/zhoul0a/Desktop/prognosis_project/check_points/infection_COVID-19/'
    models = three_way_prediction.load_models(check_point_dict, array_info)
    func_file_operation_semantic = partial(func_file_operation, model=models, batch_size=2, array_info=array_info)
    add_features.func_add_feature(top_dict_rescaled_ct, [top_dict_rescaled_ct], top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    current_fold = (0, 8)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)
    add_infection_seg('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                      '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics', fold=current_fold)
