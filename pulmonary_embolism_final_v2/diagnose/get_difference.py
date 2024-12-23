"""
get the difference array between rescaled_ct and the predicted

"""
import numpy as np
import torch.nn.functional as func
import torch.nn as nn
import analysis.center_line_and_depth_3D as center_line_and_distance
import torch


class AggregateSum(nn.Module):
    # sum the adjacent voxels to the center
    def __init__(self, kernel_size=3):
        super(AggregateSum, self).__init__()
        super().__init__()
        kernel = np.ones([1, 1, kernel_size, kernel_size, kernel_size], 'float32')
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


def difference_average_and_std(rescaled_baseline, predicted, predict_region_mask, blood_vessel_mask,
                               blood_vessel_depth=None, kernel_size=5, valid_count=None):
    """

    :param rescaled_baseline: the real ct data, like from embolism patient
    :param predicted: predicted normal vessel from our model
    :param predict_region_mask: transformer only predict certain region, this mask gives the predicted region
    :param blood_vessel_mask:
    :param blood_vessel_depth: the encoding_depth mask for blood vessel
    :param kernel_size:
    :param valid_count:
    :return:
    """
    if valid_count is None:
        valid_count = kernel_size * kernel_size

    difference_array = rescaled_baseline - predicted

    predict_region_mask = predict_region_mask * blood_vessel_mask

    difference_array = difference_array * predict_region_mask
    # the difference between baseline and prediction

    if blood_vessel_depth is None:
        blood_vessel_depth = center_line_and_distance.get_surface_distance(blood_vessel_mask)

    convolution_layer = AggregateSum(kernel_size).cuda()

    predict_region_tensor = torch.from_numpy(predict_region_mask).unsqueeze(0).unsqueeze(0).cuda()
    difference_tensor = torch.from_numpy(difference_array).unsqueeze(0).unsqueeze(0).cuda()

    voxel_counted_map = convolution_layer(predict_region_tensor) + 0.0000000001  # how many voxel is aggregated

    average_adjacent_map = convolution_layer(difference_tensor) / (voxel_counted_map + 0.0000000001)
    # the smoothed difference array

    square_bias_map = torch.square(average_adjacent_map - difference_tensor)
    std_map = convolution_layer(square_bias_map) / (torch.clip(voxel_counted_map, valid_count, None) - 1)

    std_map = std_map.to('cpu').data.numpy()[0, 0, :, :, :]
    average_adjacent_map = average_adjacent_map.to('cpu').data.numpy()[0, 0, :, :, :]
    voxel_counted_map = voxel_counted_map.to('cpu').data.numpy()[0, 0, :, :, :]

    meaningful_region = \
        predict_region_mask * np.array(blood_vessel_depth >= 4) * np.array(voxel_counted_map > valid_count)

    return average_adjacent_map, std_map, voxel_counted_map, meaningful_region


def get_log_stat_voxel_wise(average_adjacent_map, std_map, voxel_counted_map, meaningful_region):
    """

    statistic is -log(probability), here probability is the probability for the baseline higher than predicted.

    :param average_adjacent_map:
    :param std_map:
    :param voxel_counted_map:
    :param meaningful_region:
    :return:
    """


if __name__ == '__main__':
    import scipy.stats as stats
    import random

    import time
    start_time = time.time()
    temp_array = np.random.randint(0, 10, (512, 512, 512))

    significant_array = stats.norm.logsf(temp_array)

    end_time = time.time()
    print(end_time - start_time)
    print(temp_array[:3, :3, 0])
    print(significant_array[:3, :3, 0])
    exit()

    print(stats.norm.logsf(50))
    print(stats.norm.logcdf(-50))
    exit()
    value_list = [0.1, 0.2, 0.1, 0.4, 0.1, 0.7]
    print(np.mean(value_list) * np.sqrt(len(value_list)) / np.std(value_list))
    print(stats.ttest_1samp(value_list, 0))
    exit()
