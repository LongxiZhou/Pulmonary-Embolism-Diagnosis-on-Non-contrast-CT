import numpy as np
from skimage.measure import label as connect
import skimage.measure as measure
import Tool_Functions.Functions as Functions
from analysis.connect_region_detect import stat_on_connected_component


def get_key(d, value):
    return [k for k, v in d.items() if v == value]


def get_sorted_connected_region_fast(input_array, threshold=None, strict=False, show=True, sort=True):
    """
    :param input_array: the mask array, with shape [x, y, z]
    :param threshold: the threshold of cast the mask array to binary
    :param strict: whether diagonal pixel is considered as adjacent.
    :param show:
    :param sort
    :return: a dict, with key 1, 2, 3, ... (int), value is list of location: {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
             a dict, with key 1, 2, 3, ... (int), value is length(list of location)
    """
    if threshold is not None:
        input_array = np.array(input_array > threshold, 'float32')

    total_dim = len(np.shape(input_array))

    if strict:
        hops = 1
    else:
        hops = None

    labels, nums = connect(input_array, connectivity=hops, return_num=True)
    prop = measure.regionprops(labels)

    id_volume_dict = {}
    id_loc_dict = {}

    for label in range(nums):
        if prop[label].area > 0:
            id_volume_dict[label + 1] = prop[label].area
            coordinates = prop[label].coords  # (num_voxel, dim)
            location_array = []
            for dim in range(total_dim):
                location_array.append(coordinates[:, dim])
            id_loc_dict[label + 1] = Functions.get_location_list(location_array)

    if not sort:
        return id_volume_dict, id_loc_dict

    return stat_on_connected_component(id_loc_dict, show=show)


def select_region(mask, leave_count, min_volume=None, min_ratio=None, strict=False, show=True):
    """

    :param strict:
    :param mask: binary mask
    :param leave_count: num connected region leave (not strict)
    :param min_volume:
    :param min_ratio:
    :param show:
    :return: new mask
    """

    if leave_count < 1:
        raise ValueError("leave_count should no less than 1")

    new_mask = mask * 0

    if min_volume is None:
        min_volume = 0
    if min_ratio is None:
        min_ratio = 0

    id_loc_dict_sorted = get_sorted_connected_region_fast(mask, strict=strict, show=show)

    if len(id_loc_dict_sorted) == 0:
        raise ValueError("all zero array")

    included_region = 0
    max_volume = len(id_loc_dict_sorted[1])
    for key in range(1, len(id_loc_dict_sorted) + 1):
        if included_region >= leave_count:
            break
        current_component = id_loc_dict_sorted[key]
        if len(current_component) < min_ratio * max_volume:
            break
        if len(current_component) < min_volume:
            break

        new_mask[Functions.get_location_array(current_component)] = 1
        included_region += 1

    return new_mask
