
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
from functools import partial
import numpy as np

import basic_tissue_prediction.three_way_prediction as three_way_prediction
import basic_tissue_prediction.connectivity_refine as connectivity_refine
from basic_tissue_prediction.predict_rescaled import predict_blood_vessel_stage_one_rescaled_array, get_bounding_box
import analysis.get_surface_rim_adjacent_mean as get_surface


def smooth_mask(vessel_mask, surface_add=1):
    if surface_add == 0:
        return vessel_mask
    for surface in range(surface_add):
        vessel_mask = vessel_mask + get_surface.get_surface(vessel_mask, outer=True, strict=False)
    vessel_mask = vessel_mask - get_surface.get_surface(vessel_mask, outer=False, strict=False)
    return vessel_mask


def get_prediction_blood_vessel(rescaled_array, stage_one_array=None, lung_mask=None, check_point_top_dict=None,
                                batch_size=64, probability_only=False, refine_blood_vessel=True,
                                fix_ratio=True, semantic_ratio=None, probability_analysis=False):
    """
    :param probability_only: only return the stage two probability mask
    return stage two probability mask and airways vessel, blood_vessel_depth
    :param probability_analysis: if True, return stage one and stage two probability masks, and lung mask
    :param semantic_ratio: if None,  we require the airways volume is 0.08 of the lung volume, else you give a ratio.
    if ratio < 0, return the prediction_combined, which is positively correlated to the probability map.
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param stage_one_array: the probability mask in shape [512, 512, 512]
    :param lung_mask: the lung mask in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine_blood_vessel: whether use connectivity refine on airways vessels, take about 30 seconds each
    :param fix_ratio: if True, we require the airways vessel volume is 0.08 of the lung volume
    :return: the mask in shape [512, 512, 512]
    """
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        # infection and lung, and stage one data_channel is 1; stage two for tracheae, airways vessel is 1
        "enhanced_channel": 2,
        # infection and lung, stage one, enhance_channel is 0; stage two for tracheae, airways vessel is 2
        "window": (-1, 0, 1),  # infection, lung, window is (-5, -2, 0, 2, 5); tracheae and airways vessel (-1, 0, 1)
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    ratio_high = 0.108  # for high recall enhance channel
    ratio_low = 0.043  # for high precision enhance channel
    if fix_ratio:
        ratio_semantic = 0.075  # we require the blood vessel volume is 0.075 of the lung volume
    else:
        ratio_semantic = None
    check_point_top_dict = '/home/zhoul0a/Desktop/prognosis_project/check_points/'
    print("check_point_top_dict:", check_point_top_dict)
    if stage_one_array is None:
        stage_one_array = predict_blood_vessel_stage_one_rescaled_array(
            rescaled_array, check_point_top_dict, batch_size)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_bounding_box(lung_mask, pad=0)
    valid_mask = np.zeros(np.shape(stage_one_array), 'float32')
    valid_mask[x_min: x_max, y_min: y_max, z_min: z_max] = 1
    stage_one_array = stage_one_array * valid_mask

    check_point_dict = os.path.join(check_point_top_dict, 'blood_vessel_seg_stage_two/')
    prediction_combined = three_way_prediction.three_way_predict_stage_two(
        rescaled_array, stage_one_array, lung_mask, ratio_low, ratio_high,
        check_point_dict, array_info, None, batch_size)

    prediction_combined = prediction_combined * valid_mask

    if probability_only:
        return prediction_combined/3

    if probability_analysis:
        return stage_one_array/3, prediction_combined/3, lung_mask

    if semantic_ratio is not None:
        ratio_semantic = semantic_ratio
    if ratio_semantic is not None and ratio_semantic < 0:
        return prediction_combined
    if ratio_semantic is not None:
        prediction = three_way_prediction.get_top_rated_points(lung_mask, prediction_combined, ratio_semantic)
    else:
        prediction = np.array(prediction_combined > 2., 'float32')
    if refine_blood_vessel:
        prediction = connectivity_refine.refine_mask(prediction, None, 2, lowest_ratio=0.2)

    return prediction


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, batch_size=2,
                        smooth_surface=1, refine=True):
    """

    :param smooth_surface:
    :param refine: connectivity refine
    :param list_top_dict_reference: [top_dict_for_file_names]
    :param dataset_sub_dir:
    :param file_name:
    :param batch_size
    :return: denoise rescaled ct, in [512, 512, 512]
    """

    file_path = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    file_path_lung = os.path.join(
        Functions.get_father_dict(list_top_dict_reference[0]), dataset_sub_dir,
        'semantics/lung_mask', file_name[:-4] + '.npz')
    print("loading:", file_path)

    if file_path[-1] == 'z':
        assert file_path[:-4] + '.npz' == file_path
        rescaled_ct = np.load(file_path)['array']
    else:
        assert file_path[:-4] + '.npy' == file_path
        rescaled_ct = np.load(file_path)

    lung_mask = np.array(np.load(file_path_lung)['array'], 'float32')

    vessel_mask = get_prediction_blood_vessel(rescaled_ct, None, lung_mask, None, batch_size, False,
                                              refine)

    vessel_mask = smooth_mask(vessel_mask, smooth_surface)

    return vessel_mask


def func_file_save(save_dict, file_name, feature_package):
    Functions.save_np_array(os.path.join(save_dict, 'blood_mask'), file_name[:-4] + '.npz', feature_package,
                            compress=True, dtype='float16')


def func_check_processed(save_dict, file_name):
    path_saved = os.path.join(save_dict, 'blood_mask', file_name[:-4] + '.npz')
    if os.path.exists(path_saved):
        return True
    return False


def add_blood_vessel_seg(top_dict_rescaled_ct, top_dict_save, fold=(0, 1)):
    func_file_operation_semantic = partial(func_file_operation, batch_size=2,
                                           smooth_surface=1, refine=True)
    add_features.func_add_feature(top_dict_rescaled_ct, [top_dict_rescaled_ct], top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    current_fold = (0, 8)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_fold[0] % 2)
    add_blood_vessel_seg('/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/',
                         '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics/', fold=current_fold)
