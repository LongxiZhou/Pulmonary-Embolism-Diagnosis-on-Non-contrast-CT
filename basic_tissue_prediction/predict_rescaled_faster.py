"""
input: a rescaled array in shape [512, 512, 512]
check_point_top_dict: a package containing all_file models: 'air_way_seg_stage_one', 'blood_vessel_seg_stage_one',
'air_way_seg_stage_two', 'blood_vessel_seg_stage_two', 'lung_seg', 'infection_COVID-19';
each model_guided has three directions: 'X', 'Y', 'Z'; each direction has one check point.
output: varies segmentation
"""
import numpy as np
import os
from Tool_Functions.Functions import get_bounding_box
import basic_tissue_prediction.three_way_prediction as three_way_prediction
import basic_tissue_prediction.connectivity_refine as connectivity_refine
import warnings
ibex = False
if not ibex:
    top_directory_check_point = '/home/zhoul0a/Desktop/prognosis_project/check_points/'
    # where the checkpoints stored: top_directory_check_point/model_type/direction/best_model-direction.pth
else:
    top_directory_check_point = '/ibex/scratch/projects/c2052/prognosis_project/'

array_info = {
    "resolution": (1, 1, 1),
    "data_channel": 1,  # infection and lung, and stage one data_channel is 1; stage two for tracheae, airways vessel is 1
    "enhanced_channel": 2,  # infection and lung, stage one, enhance_channel is 0; stage two for tracheae, airways vessel
    # is 2
    "window": (-1, 0, 1),  # infection, lung, window is (-5, -2, 0, 2, 5); tracheae and airways vessel (-1, 0, 1)
    "positive_semantic_channel": None,  # prediction phase this should be None
    "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
    "mute_output": True,  # if you want to see prediction details, set is as False
    "wrong_scan": None,
    "init_features": 16
}


def get_prediction_rescaled_array_binary(rescaled_array, check_point_dict, threshold=None, view_weight=None, batch_size=64):
    """
    :param view_weight: None means three view considered the same. Otherwise, a tuple, like (1, 2, 0), means X is twice
    important than Y, and do not take Z into account. Note, weight will be rescaled into sum(view_weight) = 3
    :param rescaled_array: numpy array in shape like [512, 512, 512]
    :param check_point_dict: where the model_guided saved, should in check_point_dict/direction/model_name.pth
    :param threshold: the threshold for the three way prediction, should in (0, 3), or None. None means return the sum
    of the probability map of three directions.
    :param batch_size: the batch_size when prediction
    :return: the prediction
    """
    assert len(os.listdir(check_point_dict)) == 3
    for direction in os.listdir(check_point_dict):
        assert len(os.listdir(os.path.join(check_point_dict, direction))) == 1
    print("check_point_dict:", check_point_dict)
    prediction = three_way_prediction.three_way_predict_binary_class(rescaled_array, check_point_dict, array_info, view_weight,
                                                                     threshold, batch_size)
    return prediction


def get_prediction_rescaled_array_multi_class(rescaled_array, check_point_dict, threshold=None, view_weight=None,
                                              batch_size=64):
    """
    return a array with shape [:, :, :, num_semantics], first channel is negative, then positive channel 1, 2, ...
    Warning: if use threshold, np.sum(prediction, axis_list=3) may not equal to np.ones
    :param view_weight: None means three view considered the same. Otherwise, a tuple, like (1, 2, 0), means X is twice
    important than Y, and do not take Z into account. Note, weight will be rescaled into sum(view_weight) = 3
    :param rescaled_array: numpy array in shape like [512, 512, 512]
    :param check_point_dict: where the model_guided saved, should in check_point_dict/direction/model_name.pth
    :param threshold: the threshold for the three way prediction, should in (0, 3), or None. None means return the sum
    of the probability map of three directions.
    :param batch_size: the batch_size when prediction
    :return: the prediction
    """
    assert len(os.listdir(check_point_dict)) == 3
    for direction in os.listdir(check_point_dict):
        assert len(os.listdir(os.path.join(check_point_dict, direction))) == 1
    print("check_point_dict:", check_point_dict)
    prediction = three_way_prediction.three_way_predict_multi_class(rescaled_array, check_point_dict, array_info, view_weight,
                                                                    threshold, batch_size)
    return prediction


def predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict=None, batch_size=64, refine=False,
                                      threshold=2.):
    """
    :param threshold:
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_top_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine: whether refine lung mask. refine lung will take about 5 min each scan.
    :return: lung mask for the rescaled array, binary numpy array in shape [512, 512, 512], 0 outer lung 1 inner lung.
    """
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
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
    print("predicting lung masks\n")
    check_point_directory = os.path.join(check_point_top_dict, 'lung_seg/')
    lung_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=threshold,
                                                     batch_size=batch_size)
    if not refine:
        array_info = original_array_info
        return lung_mask
    else:
        lung_mask = connectivity_refine.refine_mask(lung_mask, None, 2, lowest_ratio=0.1)
        array_info = original_array_info
        return lung_mask


def predict_heart_rescaled_array(rescaled_array, check_point_top_dict=None, threshold=2.3, batch_size=64, refine=False):
    """
    :param threshold:
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_top_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine: whether refine lung mask. refine lung will take about 5 min each scan.
    :return: lung mask for the rescaled array, binary numpy array in shape [512, 512, 512], 0 outer lung 1 inner lung.
    """
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    print("predicting whole heart\n")
    check_point_directory = os.path.join(check_point_top_dict, 'heart_seg/')
    heart_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=threshold,
                                                      view_weight=(1, 1, 1), batch_size=batch_size)
    if threshold is None:
        assert refine is False
        return heart_mask
    if not refine:
        array_info = original_array_info
        return heart_mask
    else:
        heart_mask = connectivity_refine.refine_mask(heart_mask, None, 1)
        array_info = original_array_info
        return heart_mask


def predict_pulmonary_nodules_rescaled_array(rescaled_array, check_point_top_dict=None, threshold=2, batch_size=64,
                                             refine=False):
    """
    :param threshold:
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_top_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine: whether refine lung mask. refine lung will take about 5 min each scan.
    :return: lung mask for the rescaled array, binary numpy array in shape [512, 512, 512], 0 outer lung 1 inner lung.
    """
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-5, -2, -1, 0, 1, 2, 5),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    print("predicting pulmonary nodules\n")
    check_point_directory = os.path.join(check_point_top_dict, 'pulmonary_nodule_v1/')
    nodule_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=threshold,
                                                      view_weight=(1, 1, 1), batch_size=batch_size)
    if threshold is None:
        assert refine is False
        return nodule_mask
    if not refine:
        array_info = original_array_info
        return nodule_mask
    else:
        heart_mask = connectivity_refine.refine_mask(nodule_mask, None, 7)
        array_info = original_array_info
        return heart_mask


def predict_covid_19_infection_rescaled_array(rescaled_array, check_point_top_dict=None, lung_mask=None, batch_size=64, threshold=2.):
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    if lung_mask is None:
        lung_mask = predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, False)
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
    print("predicting covid 19 infection\n")
    check_point_directory = os.path.join(check_point_top_dict, 'infection_COVID-19/')
    infection_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=threshold,
                                                          batch_size=batch_size)
    infection_mask = infection_mask * lung_mask
    array_info = original_array_info
    return infection_mask


def predict_airway_stage_one_rescaled_array(rescaled_array, check_point_top_dict=None, batch_size=64):
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    print("predicting air-way stage one\n")
    check_point_directory = os.path.join(check_point_top_dict, 'air_way_seg_stage_one/')
    stage_one_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=None,
                                                          batch_size=batch_size)
    array_info = original_array_info
    return stage_one_mask


def predict_blood_vessel_stage_one_rescaled_array(rescaled_array, check_point_top_dict=None, batch_size=64):
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    print("predicting blood_vessel stage one\n")
    check_point_directory = os.path.join(check_point_top_dict, 'blood_vessel_seg_stage_one/')
    stage_one_mask = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=None,
                                                          batch_size=batch_size)
    array_info = original_array_info
    return stage_one_mask


def get_prediction_airway(rescaled_array, stage_one_array=None, lung_mask=None, check_point_top_dict=None,
                          batch_size=64, refine_lung=False, refine_airway=True, fix_ratio=True, semantic_ratio=None):
    """
    :param semantic_ratio: if None,  we require the airway volume is 0.018 of the lung volume, else you give a ratio.
    if ratio < 0, return the prediction_combined, which is positively correlated to the probability map.
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param stage_one_array: the probability mask in shape [512, 512, 512]
    :param lung_mask: the lung mask in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine_lung: whether use connectivity refine on lung mask, take about 5 min each
    :param refine_airway: whether use connectivity refine on airways, take about 30 seconds each
    :param fix_ratio: if True, we require the airway volume is 0.018 of the lung volume
    :return: the mask in shape [512, 512, 512]
    """
    global array_info
    original_array_info = array_info
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
    ratio_high = 0.023  # for high recall enhance channel
    ratio_low = 0.0066  # for high precision enhance channel
    if fix_ratio:
        ratio_semantic = 0.016  # we require the airway volume is 0.018 of the lung volume
    else:
        ratio_semantic = None
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    print("check_point_top_dict:", check_point_top_dict)
    if lung_mask is None:
        lung_mask = predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, refine_lung)
    if stage_one_array is None:
        stage_one_array = predict_airway_stage_one_rescaled_array(rescaled_array, check_point_top_dict, batch_size)
    check_point_dict = os.path.join(check_point_top_dict, 'air_way_seg_stage_two/')
    prediction_combined = three_way_prediction.three_way_predict_stage_two(rescaled_array, stage_one_array, lung_mask,
                                                                           ratio_low, ratio_high, check_point_dict, array_info, None, batch_size)
    if semantic_ratio is not None:
        ratio_semantic = semantic_ratio
    if ratio_semantic is not None and ratio_semantic < 0:
        return prediction_combined
    if ratio_semantic is not None:
        prediction = three_way_prediction.get_top_rated_points(lung_mask, prediction_combined, ratio_semantic)
    else:
        prediction = np.array(prediction_combined > 2., 'float32')
    if refine_airway:
        prediction = connectivity_refine.refine_mask(prediction, None, 7)
    array_info = original_array_info
    return prediction


def get_prediction_blood_vessel(rescaled_array, stage_one_array=None, lung_mask=None, check_point_top_dict=None,
                                batch_size=64, probability_only=False, refine_lung=False, refine_blood_vessel=True,
                                fix_ratio=True, semantic_ratio=None, probability_analysis=False, artery_vein=False):
    """
    :param probability_only: only return the stage two probability mask
    :param artery_vein: this is for output results needed for artery_vein_discrimination,
    return stage two probability mask and airways vessel, blood_vessel_depth
    :param probability_analysis: if True, return stage one and stage two probability masks, and lung mask
    :param semantic_ratio: if None,  we require the airways volume is 0.08 of the lung volume, else you give a ratio.
    if ratio < 0, return the prediction_combined, which is positively correlated to the probability map.
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param stage_one_array: the probability mask in shape [512, 512, 512]
    :param lung_mask: the lung mask in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine_lung: whether use connectivity refine on lung mask, take about 5 min each
    :param refine_blood_vessel: whether use connectivity refine on airways vessels, take about 30 seconds each
    :param fix_ratio: if True, we require the airways vessel volume is 0.08 of the lung volume
    :return: the mask in shape [512, 512, 512]
    """
    global array_info
    original_array_info = array_info
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
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    print("check_point_top_dict:", check_point_top_dict)
    if lung_mask is None:
        lung_mask = predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, refine_lung)
    if stage_one_array is None:
        stage_one_array = predict_blood_vessel_stage_one_rescaled_array(rescaled_array, check_point_top_dict, batch_size)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_bounding_box(lung_mask, pad=0)
    valid_mask = np.zeros(np.shape(stage_one_array), 'float32')
    valid_mask[x_min: x_max, y_min: y_max, z_min: z_max] = 1
    stage_one_array = stage_one_array * valid_mask

    check_point_dict = os.path.join(check_point_top_dict, 'blood_vessel_seg_stage_two/')
    prediction_combined = three_way_prediction.three_way_predict_stage_two(rescaled_array, stage_one_array, lung_mask,
                                                                           ratio_low, ratio_high, check_point_dict, array_info, None, batch_size)

    prediction_combined = prediction_combined * valid_mask

    array_info = original_array_info

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
        prediction = connectivity_refine.refine_mask(prediction, None, 2, lowest_ratio=0.4)

    if artery_vein:
        return prediction_combined/3, prediction

    return prediction


def get_prediction_blood_vessel_xb_as_anchor(rescaled_array, stage_one_array=None, lung_mask=None, xb_mask=None,
                                             check_point_top_dict=None, batch_size=64, probability_only=False,
                                             refine_lung=False, refine_blood_vessel=True,  fix_ratio=True,
                                             semantic_ratio=None, probability_analysis=False, artery_vein=False):
    """
    :param probability_only: only return the stage two probability mask
    :param artery_vein: this is for output results needed for artery_vein_discrimination,
    return stage two probability mask and airways vessel, blood_vessel_depth
    :param probability_analysis: if True, return stage one and stage two probability masks, and lung mask
    :param semantic_ratio: if None,  we require the airways volume is 0.08 of the lung volume, else you give a ratio.
    if ratio < 0, return the prediction_combined, which is positively correlated to the probability map.
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param stage_one_array: the probability mask in shape [512, 512, 512]
    :param lung_mask: the lung mask in shape [512, 512, 512]
    :param xb_mask: the heart mask in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine_lung: whether use connectivity refine on lung mask, take about 5 min each
    :param refine_blood_vessel: whether use connectivity refine on airways vessels, take about 30 seconds each
    :param fix_ratio: if True, we require the airways vessel volume is 0.08 of the lung volume
    :return: the mask in shape [512, 512, 512]
    """
    global array_info
    original_array_info = array_info
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

    ratio_high = 0.703  # for high recall enhance channel
    ratio_low = 0.2748  # for high precision enhance channel
    if fix_ratio:
        ratio_semantic = 0.41
    else:
        ratio_semantic = None
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    print("check_point_top_dict:", check_point_top_dict)
    if lung_mask is None:
        lung_mask = predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, refine_lung)
    if xb_mask is None:
        xb_mask = predict_heart_rescaled_array(rescaled_array)
    if stage_one_array is None:
        stage_one_array = predict_blood_vessel_stage_one_rescaled_array(rescaled_array, check_point_top_dict, batch_size)
    check_point_dict = os.path.join(check_point_top_dict, 'blood_vessel_seg_stage_two/')

    enhanced_channel_one = get_top_rated_points_use_xb_as_anchor(xb_mask, lung_mask, stage_one_array, ratio_high)
    enhanced_channel_two = get_top_rated_points_use_xb_as_anchor(xb_mask, lung_mask, stage_one_array, ratio_low)
    enhance_channels = (enhanced_channel_one, enhanced_channel_two)

    prediction_combined = \
        three_way_prediction.three_way_predict_stage_two(rescaled_array, stage_one_array, lung_mask, ratio_low,
                                                         ratio_high, check_point_dict, array_info, None, batch_size,
                                                         enhance_channels)
    array_info = original_array_info

    if probability_only:
        return prediction_combined/3

    if probability_analysis:
        return stage_one_array/3, prediction_combined/3, lung_mask

    if semantic_ratio is not None:
        ratio_semantic = semantic_ratio
    if ratio_semantic is not None and ratio_semantic < 0:
        return prediction_combined
    if ratio_semantic is not None:
        prediction = get_top_rated_points_use_xb_as_anchor(xb_mask, lung_mask, prediction_combined, ratio_semantic)
    else:
        prediction = np.array(prediction_combined > 2., 'float32')
    if refine_blood_vessel:
        prediction = connectivity_refine.refine_mask(prediction, None, 2, lowest_ratio=0.4)

    if artery_vein:
        return prediction_combined/3, prediction

    return prediction


def get_predict_blood_vessel_root(rescaled_array, lung_mask=None, blood_vessel_mask=None, check_point_top_dict=None,
                                  batch_size=64, refine_blood_vessel=False, use_ratio=True, xb_mask=None,
                                  unclear='remove'):
    """
    :param xb_mask:
    :param use_ratio: expectation for fdm_outside_lung/xb, fjm_outside_lung/xb = 0.087928414, 0.094912276
    :param blood_vessel_mask:
    :param lung_mask:
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param check_point_top_dict: where the model_guided saved, should in check_point_top_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :param refine_blood_vessel: whether refine airways vessel. refine lung will take about 20 seconds each scan.
    :param unclear: very small part of voxel (usually < 1000), are predicted as artery and vein simultaneously.
    'remove' means remove these small and unclear region, 'leave' means artery root mask and vein root mask may have
    overlap
    :return: binary numpy array in shape [512, 512, 512, 2], first channel for artery root and second for vein root
    Note, root means outside the lungs, i.e. multiplied by (1 - vessel_mask)
    """
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point

    if lung_mask is None:
        lung_mask = predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict=check_point_top_dict)
    if blood_vessel_mask is None:
        blood_vessel_mask = get_prediction_blood_vessel(rescaled_array, None, lung_mask, check_point_top_dict,
                                                        batch_size, refine_blood_vessel=refine_blood_vessel)
    enhanced_rescaled = np.zeros([512, 512, 512, 3], 'float32')
    enhanced_rescaled[:, :, :, 0] = rescaled_array
    enhanced_rescaled[:, :, :, 1] = lung_mask
    enhanced_rescaled[:, :, :, 2] = blood_vessel_mask * (1 - lung_mask)

    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 2,
        "window": (-5, -2, 0, 2, 5),
        "positive_semantic_channel": None,  # prediction phase this should be None (or only predict positive slices)
        "output_channels": 3,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }
    print("predicting blood_vessel root\n")
    check_point_directory = os.path.join(check_point_top_dict, 'blood_vessel_root/')
    if not use_ratio:
        blood_vessel_root = get_prediction_rescaled_array_multi_class(enhanced_rescaled, check_point_directory, 2.,
                                                                      batch_size=batch_size)

        blood_vessel_root[:, :, :, 1] = blood_vessel_root[:, :, :, 1] * (1 - lung_mask)
        blood_vessel_root[:, :, :, 2] = blood_vessel_root[:, :, :, 2] * (1 - lung_mask)

    else:
        blood_vessel_root = get_prediction_rescaled_array_multi_class(enhanced_rescaled, check_point_directory, None,
                                                                      batch_size=batch_size)

        blood_vessel_root[:, :, :, 1] = blood_vessel_root[:, :, :, 1] * (1 - lung_mask)
        blood_vessel_root[:, :, :, 2] = blood_vessel_root[:, :, :, 2] * (1 - lung_mask)

        if xb_mask is None:
            xb_mask = predict_heart_rescaled_array(rescaled_array, check_point_top_dict, batch_size, False)
        blood_vessel_root[:, :, :, 1] = get_top_rated_points_use_xb_as_anchor(xb_mask, lung_mask,
                                                                              blood_vessel_root[:, :, :, 1], 0.087928414)
        blood_vessel_root[:, :, :, 2] = get_top_rated_points_use_xb_as_anchor(xb_mask, lung_mask,
                                                                              blood_vessel_root[:, :, :, 2], 0.094912276)
    overlap_region = blood_vessel_root[:, :, :, 1] * blood_vessel_root[:, :, :, 2]
    overlap_ratio = np.sum(overlap_region) / np.sum(blood_vessel_root[:, :, :, 1::])
    if overlap_ratio > 0.01:
        warnings.warn("the overlap between predicted is a little bit large: overlap percentage", overlap_ratio * 100)
    if unclear == 'remove':
        blood_vessel_root[:, :, :, 1] = blood_vessel_root[:, :, :, 1] - overlap_region
        blood_vessel_root[:, :, :, 2] = blood_vessel_root[:, :, :, 2] - overlap_region
    else:
        assert unclear == 'leave'
    array_info = original_array_info
    return blood_vessel_root[:, :, :, 1::]


def predict_breast_tumor_dcm_mri(rescaled_array, check_point_top_dict=None, batch_size=2, stage_one_only=False,
                                 stage_one_mask=None):
    """
    :param stage_one_mask: give the stage one mask and start the stage two
    :param stage_one_only:
    :param rescaled_array: numpy array in shape [464, 464, 240, 3]
    :param check_point_top_dict: where the model_guided saved, should in check_point_top_dict/semantic/direction/model_name.pth
    :param batch_size: the batch_size when prediction
    :return: lung mask for the rescaled array, binary numpy array in shape [512, 512, 512], 0 outer lung 1 inner lung.
    """
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 2,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }

    if stage_one_mask is None:
        print("predicting breast tumor from DCE-MRI stage_one\n")
        check_point_directory = os.path.join(check_point_top_dict, 'breast_tumor_seg_stage_one/')
        tumor_prob = get_prediction_rescaled_array_binary(rescaled_array, check_point_directory, threshold=None,
                                                          batch_size=batch_size)
        if stage_one_only:
            return tumor_prob / 3
        std = np.std(tumor_prob)
        stage_one_mask = np.array(tumor_prob > 3 * std, 'float32')
        shape = np.shape(stage_one_mask)
        total_volume = shape[0] * shape[1] * shape[2]
        significance = 3
        max_prob = np.max(tumor_prob)
        max_num = np.sum(tumor_prob > (1 / 3) * max_prob)
        max_num = min([max_num, 0.002 * total_volume])
        while np.sum(stage_one_mask) > max_num:
            significance += 1
            stage_one_mask = np.array(tumor_prob > significance * std, 'float32')
        print("ROI: >", significance, "std")
    else:
        shape = np.shape(stage_one_mask)

    stage_two_input = np.zeros([shape[0], shape[1], shape[2], 4], 'float32')
    stage_two_input[:, :, :, 0: 3] = rescaled_array
    stage_two_input[:, :, :, 3] = stage_one_mask

    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 3,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }

    print("predicting breast tumor from DCE-MRI stage_two\n")
    check_point_directory = os.path.join(check_point_top_dict, 'breast_tumor_seg_stage_two/')
    tumor_mask = get_prediction_rescaled_array_binary(stage_two_input, check_point_directory, threshold=None,
                                                      batch_size=batch_size)
    array_info = original_array_info
    return tumor_mask


def get_top_rated_points_use_lung_as_anchor(lung_mask, prediction_combined, ratio):
    """
    :param lung_mask: npy array in float 32 and in shape [512, 512, 512]
    :param prediction_combined: sum of the probability map of the stage one
    ratio, is defined as: volume_semantic / volume lung
    :param ratio: a float like 0.043, which means we take 0.043*np.sum(mask_lung) as predicted positive.
    :return: one arrays both with shape [512, 512, 512], which is the mask of the top rated candidates
    """
    lung_pixels = np.sum(lung_mask)

    inside_lung = np.where(lung_mask > 0)
    x_min = max(np.min(inside_lung[0]), 10)
    x_max = min(np.max(inside_lung[0]), 500)
    y_min = max(np.min(inside_lung[1]), 10)
    y_max = min(np.max(inside_lung[1]), 500)
    z_min = max(np.min(inside_lung[2]), 10)
    z_max = min(np.max(inside_lung[2]), 500)

    temp_array = np.array(prediction_combined[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
    temp_array = -np.reshape(temp_array, [-1, ])
    print("getting optimal threshold...")
    temp_array = -np.sort(temp_array)

    # high precision threshold:
    print("assume that semantic volume/lung volume =", ratio)
    threshold = temp_array[int(lung_pixels * ratio)]
    print("threshold is:", threshold)

    return np.array(prediction_combined > threshold, 'float32')


def get_top_rated_points_use_xb_as_anchor(xb, lung_mask, prediction_combined, ratio):
    """
    :param xb: the heart seg
    :param lung_mask: npy array in float 32 and in shape [512, 512, 512]
    :param prediction_combined: sum of the probability map of the stage one
    ratio, is defined as: volume_semantic / volume lung
    :param ratio: a float like 0.043, which means we take 0.043*np.sum(xb) as predicted positive.
    :return: one arrays both with shape [512, 512, 512], which is the mask of the top rated candidates
    """
    xb_pixels = np.sum(xb)

    inside_lung = np.where(lung_mask > 0)
    x_min = max(np.min(inside_lung[0]), 10)
    x_max = min(np.max(inside_lung[0]), 500)
    y_min = max(np.min(inside_lung[1]), 10)
    y_max = min(np.max(inside_lung[1]), 500)
    z_min = max(np.min(inside_lung[2]), 10)
    z_max = min(np.max(inside_lung[2]), 500)

    temp_array = np.array(prediction_combined[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
    temp_array = -np.reshape(temp_array, [-1, ])
    print("getting optimal threshold...")
    temp_array = -np.sort(temp_array)

    # high precision threshold:
    print("assume that semantic volume/heart volume =", ratio)
    threshold = temp_array[int(xb_pixels * ratio)]
    print("threshold is:", threshold)

    return np.array(prediction_combined > threshold, 'float32')
