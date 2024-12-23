import os
import sys
import numpy as np

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import models.Unet_2D.test as test_model


def check_point_path_generator(check_point_dict, direction):
    """
    :param check_point_dict: directory where the check_points, like: '/ibex/check_points/'. in this dict, there should
    be three sub-dict: X/, Y/, Z/, which means the directions. in each direction, there is one model_guided.
    :param direction: one of 'X', 'Y', 'Z'.
    :return: freq path of the correct check_point_file
    """
    sub_dict = os.path.join(check_point_dict, direction)
    model_name = os.listdir(sub_dict)
    assert len(model_name) == 1
    model_path = os.path.join(sub_dict, model_name[0])
    return model_path


def load_models(check_point_dict, array_info):
    check_point_path_x = check_point_path_generator(check_point_dict, 'X')
    check_point_path_y = check_point_path_generator(check_point_dict, 'Y')
    check_point_path_z = check_point_path_generator(check_point_dict, 'Z')

    loaded_model_x = test_model.load_model(check_point_path_x, array_info)
    loaded_model_y = test_model.load_model(check_point_path_y, array_info)
    loaded_model_z = test_model.load_model(check_point_path_z, array_info)

    return loaded_model_x, loaded_model_y, loaded_model_z


def three_way_predict_binary_class(rescaled_array, check_point_dict, array_info,
                                   view_weight=None, threshold=2., batch_size=64, loaded_models=None):
    """
    :param loaded_models:
    :param view_weight: None means three view considered the same. Otherwise, freq tuple, like (1, 2, 0), means X is twice
    important than Y, and do not take Z into account. Note, weight will be rescaled into sum(view_weight) = 3
    :param array_info: contain information like "resolution", "window", "init_features" etc.
    :param rescaled_array: shape and resolution and signal normalized array
    :param batch_size: the batch_size when predicting.
    :param threshold: when combine, the threshold of considered as positive. When threshold is None, return sum of three
    directions.
    :param check_point_dict: directory check points saved, like '/ibex/scratch/projects/air_tube_seg/check_points/'
    :return: prediction mask or sum of three probability maps
    """
    if view_weight is None:
        view_weight = (1, 1, 1)
    else:
        view_weight = np.array(view_weight, 'float32')

    assert len(view_weight) == 3
    sum_weight = np.sum(view_weight)

    check_point_path_x = check_point_path_generator(check_point_dict, 'X')
    check_point_path_y = check_point_path_generator(check_point_dict, 'Y')
    check_point_path_z = check_point_path_generator(check_point_dict, 'Z')

    if loaded_models is None:
        loaded_models = load_models(check_point_dict, array_info)

    loaded_model_x, loaded_model_y, loaded_model_z = loaded_models

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_x, array_info,
                                                            batch_size=batch_size, model=loaded_model_x)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_y, array_info,
                                                            batch_size=batch_size, model=loaded_model_y)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_z, array_info,
                                                            batch_size=batch_size, model=loaded_model_z)
    if threshold is not None:
        prediction = np.array(prediction_X * view_weight[0] + prediction_Y * view_weight[1] +
                              prediction_Z * view_weight[2] > (threshold * sum_weight / 3), 'float32')
    else:
        return (prediction_X * view_weight[0] + prediction_Y * view_weight[1] + prediction_Z * view_weight[2]) \
               * 3 / sum_weight
    return prediction


def three_way_predict_binary_class_faster(rescaled_array, view_weight=None, threshold=2.,
                                          batch_size=2, loaded_models=None, array_info=None):
    """
    :param array_info:
    :param loaded_models:
    :param view_weight: None means three view considered the same. Otherwise, freq tuple, like (1, 2, 0), means X is twice
    important than Y, and do not take Z into account. Note, weight will be rescaled into sum(view_weight) = 3
    :param rescaled_array: shape and resolution and signal normalized array
    :param batch_size: the batch_size when predicting.
    :param threshold: when combine, the threshold of considered as positive. When threshold is None, return sum of three
    directions.
    :return: prediction mask or sum of three probability maps
    """
    if view_weight is None:
        view_weight = (1, 1, 1)
    else:
        view_weight = np.array(view_weight, 'float32')

    assert len(view_weight) == 3
    sum_weight = np.sum(view_weight)

    assert loaded_models is not None

    loaded_model_x, loaded_model_y, loaded_model_z = loaded_models

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', None, array_info,
                                                            batch_size=batch_size, model=loaded_model_x)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', None, array_info,
                                                            batch_size=batch_size, model=loaded_model_y)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', None, array_info,
                                                            batch_size=batch_size, model=loaded_model_z)
    prediction = prediction_X * view_weight[0] + prediction_Y * view_weight[1] + prediction_Z * view_weight[2]

    del prediction_X, prediction_Y, prediction_Z

    if threshold is not None:
        prediction = np.array(prediction > (threshold * sum_weight / 3), 'float32')
    else:
        return prediction * 3 / sum_weight

    return prediction


def three_way_predict_multi_class(rescaled_array, check_point_dict, array_info, view_weight=None, threshold=2., batch_size=64):
    """
    :param view_weight: None means three view considered the same. Otherwise, freq tuple, like (1, 2, 0), means X is twice
    important than Y, and do not take Z into account. Note, weight will be rescaled into sum(view_weight) = 3
    :param array_info: contain information like "resolution", "window", "init_features" etc.
    :param rescaled_array: shape and resolution and signal normalized array
    :param batch_size: the batch_size when predicting.
    :param threshold: when combine, the threshold of considered as positive. When threshold is None, return sum of three
    directions.
    :param check_point_dict: directory check points saved, like '/ibex/scratch/projects/air_tube_seg/check_points/'
    :return: prediction mask or sum of three probability maps
    """
    if view_weight is None:
        view_weight = (1, 1, 1)
    else:
        view_weight = np.array(view_weight, 'float32')

    assert len(view_weight) == 3
    sum_weight = np.sum(view_weight)

    check_point_path_X = check_point_path_generator(check_point_dict, 'X')
    check_point_path_Y = check_point_path_generator(check_point_dict, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_dict, 'Z')

    prediction_X = test_model.predict_one_scan_multi_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_multi_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_multi_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                            evaluate=False, batch_size=batch_size)

    if threshold is not None:
        prediction = np.array(prediction_X * view_weight[0] + prediction_Y * view_weight[1] +
                              prediction_Z * view_weight[2] > (threshold * sum_weight / 3), 'float32')
    else:
        return (prediction_X * view_weight[0] + prediction_Y * view_weight[1] + prediction_Z * view_weight[2]) \
               * 3 / sum_weight
    return prediction


def get_enhance_channel(lung_mask, stage_one_prediction, ratio_low, ratio_high):
    """
    :param lung_mask: npy array in float 32 and in shape [512, 512, 512]
    :param stage_one_prediction: sum of the probability map of the stage one
    ratio, is defined as: volume_semantic / volume lung
    :param ratio_low: freq float like 0.043, which means we want to reach high precision, only take 0.043*np.sum(mask_lung)
    as predicted positive.
    :param ratio_high: freq float like 0.108, which means we want to reach high recall, take up to 0.108*np.sum(mask_lung)
    as predicted positive
    :return: two arrays both with shape [512, 512, 512]. one is the high recall mask, the other is high precision mask
    """
    lung_pixels = np.sum(lung_mask)
    print("there are:", lung_pixels, "of lung voxels")

    inside_lung = np.where(lung_mask > 0)
    x_min = max(np.min(inside_lung[0]), 10)
    x_max = min(np.max(inside_lung[0]), 500)
    y_min = max(np.min(inside_lung[1]), 10)
    y_max = min(np.max(inside_lung[1]), 500)
    z_min = max(np.min(inside_lung[2]), 10)
    z_max = min(np.max(inside_lung[2]), 500)

    lung_range = (x_min, x_max, y_min, y_max, z_min, z_max)
    print("lung range (x_min, x_max, y_min, y_max, z_min, z_max):", lung_range)

    prediction_combined = stage_one_prediction

    temp_array = np.array(prediction_combined[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
    temp_array = -np.reshape(temp_array, [-1, ])
    print("getting optimal threshold...")
    temp_array = -np.sort(temp_array)

    # high precision threshold:
    print("high precision threshold assume that semantic volume/lung volume =", ratio_low)
    threshold_precision = temp_array[int(lung_pixels * ratio_low)]
    print("threshold for high precision:", threshold_precision)

    # high recall threshold:
    print("high recall threshold assume that semantic volume/lung volume =", ratio_high)
    threshold_recall = temp_array[int(lung_pixels * ratio_high)]
    print("threshold for high recall:", threshold_recall)

    return np.array(prediction_combined > threshold_recall, 'float32'), \
           np.array(prediction_combined > threshold_precision, 'float32')


def get_top_rated_points(lung_mask, prediction_combined, ratio):
    """
    :param lung_mask: npy array in float 32 and in shape [512, 512, 512]
    :param prediction_combined: sum of the probability map of the stage one
    ratio, is defined as: volume_semantic / volume lung
    :param ratio: freq float like 0.043, which means we take 0.043*np.sum(mask_lung) as predicted positive.
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


def three_way_predict_stage_two(rescaled_array, stage_one_prediction, lung_mask, ratio_low, ratio_high,
                                check_point_dict, array_info, threshold=2., batch_size=64, enhance_channel=None):
    """
    :param lung_mask: the lung mask of the rescaled array, binary, with shape [512, 512, 512]
    :param ratio_low: ratio of the positive semantic to the lung region, for high precision channel
    :param ratio_high: ratio of the positive semantic to the lung region, for high recall channel
    :param stage_one_prediction: sum of the probability map of the stage one
    :param array_info: contain information like "resolution", "window", "init_features" etc.
    :param rescaled_array: shape and resolution and signal normalized array
    :param batch_size: the batch_size when predicting.
    :param enhance_channel: direct give enhance channel (array_shaped(512, 512, 512), array_shaped(512, 512, 512))
    :param threshold: when combine, the threshold of considered as positive. When threshold is None, return sum of three
    directions.
    :param check_point_dict: directory check points saved, like '/ibex/scratch/projects/air_tube_seg/check_points/'
    :return: prediction mask or sum of three probability maps
    """

    if enhance_channel is None:
        enhanced_channel_one, enhanced_channel_two = get_enhance_channel(lung_mask, stage_one_prediction, ratio_low,
                                                                         ratio_high)
    else:
        enhanced_channel_one, enhanced_channel_two = enhance_channel

    shape_array = np.shape(rescaled_array)

    rescaled_array_enhance_two = np.zeros((shape_array[0], shape_array[1], shape_array[2], 3), 'float32')

    rescaled_array_enhance_two[:, :, :, 0] = rescaled_array  # data
    rescaled_array_enhance_two[:, :, :, 1] = enhanced_channel_one  # high recall
    rescaled_array_enhance_two[:, :, :, 2] = enhanced_channel_two  # high precision

    check_point_path_X = check_point_path_generator(check_point_dict, 'X')
    check_point_path_Y = check_point_path_generator(check_point_dict, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_dict, 'Z')

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'X', check_point_path_X, array_info,
                                                            batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'Y', check_point_path_Y, array_info,
                                                            batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'Z', check_point_path_Z, array_info,
                                                            batch_size=batch_size)
    if threshold is not None:
        prediction = np.array(prediction_X + prediction_Y + prediction_Z > threshold, 'float32')
    else:
        return prediction_X + prediction_Y + prediction_Z

    return prediction
