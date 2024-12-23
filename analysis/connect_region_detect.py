"""
see function name: get_sorted_connected_regions
input a 3D mask numpy array, output a dict, with key 1, 2, 3, ... (int), which conforms to the ranking of the volume
of the connected component. The value of the dict is lists of locations like {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
"""
import numpy as np
import Tool_Functions.Functions as Functions
import analysis.connected_region2d_and_scale_free_stat as rim_detect
import analysis.get_surface_rim_adjacent_mean as get_surface_and_rim

np.set_printoptions(precision=10, suppress=True)
epsilon = 0.001


class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


def get_connected_regions(input_array, threshold=None, strict=False, start_id=None, fast_version=True):
    """
    :param input_array: the mask array, with shape [x, y, z]
    :param threshold: the threshold of cast the mask array to binary
    :param strict: whether diagonal pixel is considered as adjacent.
    :param start_id: the connect region id
    :param fast_version
    :return: a dict, with key 1, 2, 3, ... (int), value is list of location: {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
             a dict, with key 1, 2, 3, ... (int), value is length(list of location)
             id_array has shape [a, b, c, 2], first channel is the merge count, second for region id
             optional: start_id for next stage
    """
    if threshold is not None:
        input_array = np.array(input_array > threshold, 'float32')

    if fast_version:
        assert start_id is None
        from analysis.connectivity_refine_fast import get_sorted_connected_region_fast

        return get_sorted_connected_region_fast(input_array, threshold=threshold, strict=strict, sort=False, show=False)

    shape = np.shape(input_array)
    helper_array = np.zeros([shape[0], shape[1], shape[2], 2])
    # the last dim has two channels, the first is the key, the second is the volume
    helper_array[:, :, :, 0] = -input_array
    tracheae_points = np.where(helper_array[:, :, :, 0] < -epsilon)
    num_checking_points = len(tracheae_points[0])
    # print("we will check:", num_checking_points)
    id_volume_dict = {}
    id_loc_dict = {}

    if start_id is None:
        connected_id = 1
    else:
        connected_id = start_id

    for index in range(num_checking_points):
        pixel_location = (tracheae_points[0][index], tracheae_points[1][index], tracheae_points[2][index])
        if helper_array[pixel_location[0], pixel_location[1], pixel_location[2], 0] > epsilon:
            # this means this point has been allocated id and volume
            continue
        else:
            # this means this point has been allocated id and volume
            if strict:
                volume, locations = broadcast_connected_component(helper_array, pixel_location, connected_id)
            else:
                volume, locations = broadcast_connected_component_2(helper_array, pixel_location, connected_id)
            # now, the volume and id has been broadcast to this connected component.
            id_volume_dict[connected_id] = volume
            id_loc_dict[connected_id] = locations
            connected_id += 1  # the id is 1, 2, 3, ...

    if start_id is None:
        return id_volume_dict, id_loc_dict, helper_array
    else:
        return id_volume_dict, id_loc_dict, helper_array, connected_id


def get_connected_regions_light(input_flow, strict=False):
    """
    :param input_flow: the binary mask array, with shape [x, y, z], pid_id
    :param strict: whether diagonal pixel is considered as adjacent.
    :return: a dict, with key 1, 2, 3, ... (int), value is list of location: {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
    """
    input_array = input_flow[0]
    print("processing sample_interval", input_flow[1])
    shape = np.shape(input_array)
    helper_array = np.zeros([shape[0], shape[1], shape[2], 2])
    # the last dim has two channels, the first is the key, the second is the volume
    helper_array[:, :, :, 0] = -input_array
    tracheae_points = np.where(helper_array[:, :, :, 0] < -epsilon)
    num_checking_points = len(tracheae_points[0])
    # print("we will check:", num_checking_points)
    id_loc_dict = {}
    connected_id = 1

    for index in range(num_checking_points):
        pixel_location = (tracheae_points[0][index], tracheae_points[1][index], tracheae_points[2][index])
        if helper_array[pixel_location[0], pixel_location[1], pixel_location[2], 0] > epsilon:
            # this means this point has been allocated id and volume
            continue
        else:
            # this means this point has been allocated id and volume
            if strict:
                volume, locations = broadcast_connected_component(helper_array, pixel_location, connected_id)
            else:
                volume, locations = broadcast_connected_component_2(helper_array, pixel_location, connected_id)
            # now, the volume and id has been broadcast to this connected component.
            id_loc_dict[connected_id] = locations
            connected_id += 1  # the id is 1, 2, 3, ...

    return id_loc_dict
    

def broadcast_connected_component(helper_array, initial_location, region_id):
    # id_array has shape [a, b, c, 2]
    # initial_location is a tuple, (x, y, z)
    # return the volume of this connected_component (int) and the location list like [(389, 401), (389, 402), ..].
    volume = 0  # the volume of this connected component
    un_labeled_region = [initial_location, ]
    helper_array[initial_location[0], initial_location[1], initial_location[2], 1] = region_id
    region_locations = []
    while un_labeled_region:  # this mean un_labeled_region is not empty
        location = un_labeled_region.pop()

        region_locations.append(location)  # get the locations of the connected component
        volume += 1

        if helper_array[location[0] + 1, location[1], location[2], 0] < -epsilon:
            # whether the adjacent pixel is in the same connected_component
            if not helper_array[location[0] + 1, location[1], location[2], 1] == region_id:
                # this adjacent location is not visited
                un_labeled_region.append((location[0] + 1, location[1], location[2]))
                helper_array[location[0] + 1, location[1], location[2], 1] = region_id  # label this unlabeled pixel
        if helper_array[location[0] - 1, location[1], location[2], 0] < -epsilon:
            if not helper_array[location[0] - 1, location[1], location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2]))
                helper_array[location[0] - 1, location[1], location[2], 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2], 0] < -epsilon:
            if not helper_array[location[0], location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2]))
                helper_array[location[0], location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0], location[1] - 1, location[2], 0] < -epsilon:
            if not helper_array[location[0], location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2]))
                helper_array[location[0], location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0], location[1], location[2] + 1, 0] < -epsilon:
            if not helper_array[location[0], location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] + 1))
                helper_array[location[0], location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1], location[2] - 1, 0] < -epsilon:
            if not helper_array[location[0], location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] - 1))
                helper_array[location[0], location[1], location[2] - 1, 1] = region_id

    for location in region_locations:
        helper_array[location[0], location[1], location[2], 0] = volume
    # print('this component has id', region_id, 'volume', volume)
    return volume, region_locations


def broadcast_connected_component_2(helper_array, initial_location, region_id):
    # the difference is that here diagonal pixels are considered as adjacency.
    # id_array has shape [a, b, c, 2]
    # initial_location is a tuple, (x, y, z)
    # return the volume of this connected_component (int) and the location list like [(389, 401), (389, 402), ..].
    volume = 0  # the volume of this connected component
    un_labeled_region = [initial_location, ]
    helper_array[initial_location[0], initial_location[1], initial_location[2], 1] = region_id
    region_locations = []
    while un_labeled_region:  # this mean un_labeled_region is not empty
        location = un_labeled_region.pop()

        region_locations.append(location)  # get the locations of the connected component
        volume += 1
        if not np.min(helper_array[location[0]-1:location[0]+2, location[1]-1:location[1]+2,
                      location[2]-1:location[2]+2]) < -epsilon:
            continue

        if helper_array[location[0] + 1, location[1], location[2], 0] < -epsilon:  # (1, 0, 0)
            # whether the adjacent pixel is in the same connected_component
            if not helper_array[location[0] + 1, location[1], location[2], 1] == region_id:
                # this adjacent location is not visited
                un_labeled_region.append((location[0] + 1, location[1], location[2]))
                helper_array[location[0] + 1, location[1], location[2], 1] = region_id  # label this unlabeled pixel
        if helper_array[location[0] - 1, location[1], location[2], 0] < -epsilon:
            if not helper_array[location[0] - 1, location[1], location[2], 1] == region_id:  # (-1, 0, 0)
                un_labeled_region.append((location[0] - 1, location[1], location[2]))
                helper_array[location[0] - 1, location[1], location[2], 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2], 0] < -epsilon:  # (0, 1, 0)
            if not helper_array[location[0], location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2]))
                helper_array[location[0], location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0], location[1] - 1, location[2], 0] < -epsilon:  # (0, -1, 0)
            if not helper_array[location[0], location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2]))
                helper_array[location[0], location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0], location[1], location[2] + 1, 0] < -epsilon:  # (0, 0, 1)
            if not helper_array[location[0], location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] + 1))
                helper_array[location[0], location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1], location[2] - 1, 0] < -epsilon:  # (0, 0, -1)
            if not helper_array[location[0], location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] - 1))
                helper_array[location[0], location[1], location[2] - 1, 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2], 0] < -epsilon:  # (-1, -1, 0)
            if not helper_array[location[0] - 1, location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2]))
                helper_array[location[0] - 1, location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2], 0] < -epsilon:  # (-1, 1, 0)
            if not helper_array[location[0] - 1, location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2]))
                helper_array[location[0] - 1, location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2], 0] < -epsilon:  # (1, 1, 0)
            if not helper_array[location[0] + 1, location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2]))
                helper_array[location[0] + 1, location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2], 0] < -epsilon:  # (1, -1, 0)
            if not helper_array[location[0] + 1, location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2]))
                helper_array[location[0] + 1, location[1] - 1, location[2], 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 0] < -epsilon:  # (-1, -1, 1)
            if not helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2] + 1))
                helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 0] < -epsilon:  # (-1, 1, 1)
            if not helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2] + 1))
                helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 0] < -epsilon:  # (1, 1, 1)
            if not helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2] + 1))
                helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 0] < -epsilon:  # (1, -1, 1)
            if not helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2] + 1))
                helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 1] = region_id

        if helper_array[location[0], location[1] - 1, location[2] + 1, 0] < -epsilon:  # (0, -1, 1)
            if not helper_array[location[0], location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2] + 1))
                helper_array[location[0], location[1] - 1, location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2] + 1, 0] < -epsilon:  # (0, 1, 1)
            if not helper_array[location[0], location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2] + 1))
                helper_array[location[0], location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1], location[2] + 1, 0] < -epsilon:  # (1, 0, 1)
            if not helper_array[location[0] + 1, location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1], location[2] + 1))
                helper_array[location[0] + 1, location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0] - 1, location[1], location[2] + 1, 0] < -epsilon:  # (-1, 0, 1)
            if not helper_array[location[0] - 1, location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2] + 1))
                helper_array[location[0] - 1, location[1], location[2] + 1, 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 0] < -epsilon:  # (-1, -1, -1)
            if not helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2] - 1))
                helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 0] < -epsilon:  # (-1, 1, -1)
            if not helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2] - 1))
                helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 0] < -epsilon:  # (1, 1, -1)
            if not helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2] - 1))
                helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 0] < -epsilon:  # (1, -1, -1)
            if not helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2] - 1))
                helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 1] = region_id

        if helper_array[location[0], location[1] - 1, location[2] - 1, 0] < -epsilon:  # (0, -1, -1)
            if not helper_array[location[0], location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2] - 1))
                helper_array[location[0], location[1] - 1, location[2] - 1, 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2] - 1, 0] < -epsilon:  # (0, 1, -1)
            if not helper_array[location[0], location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2] - 1))
                helper_array[location[0], location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1], location[2] - 1, 0] < -epsilon:  # (1, 0, -1)
            if not helper_array[location[0] + 1, location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1], location[2] - 1))
                helper_array[location[0] + 1, location[1], location[2] - 1, 1] = region_id
        if helper_array[location[0] - 1, location[1], location[2] - 1, 0] < -epsilon:  # (-1, 0, -1)
            if not helper_array[location[0] - 1, location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2] - 1))
                helper_array[location[0] - 1, location[1], location[2] - 1, 1] = region_id

    for location in region_locations:
        helper_array[location[0], location[1], location[2], 0] = volume
    # print('this component has id', region_id, 'volume', volume)
    return volume, region_locations


def sort_on_id_loc_dict(id_loc_dict, id_volume_dict=None):
    # refactor the key of the connected_components
    keys_list = list(id_loc_dict.keys())
    number_keys = len(keys_list)
    if id_volume_dict is None:
        id_volume_dict = {}
        for i in range(1, number_keys + 1):
            id_volume_dict[i] = len(id_loc_dict[i])
    old_factor_list = []
    for i in range(1, number_keys + 1):
        old_factor_list.append((i, id_volume_dict[i]))

    def adjacency_cmp(tuple_a, tuple_b):
        return tuple_a[1] - tuple_b[1]

    from functools import cmp_to_key
    old_factor_list.sort(key=cmp_to_key(adjacency_cmp), reverse=True)

    id_loc_dict_sorted = {}
    id_volume_dict_sorted = {}
    for i in range(0, number_keys):
        id_loc_dict_sorted[i + 1] = id_loc_dict[old_factor_list[i][0]]
        id_volume_dict_sorted[i + 1] = id_volume_dict[old_factor_list[i][0]]
    return id_loc_dict_sorted, id_volume_dict_sorted


def stat_on_connected_component(id_loc_dict, total_volume=None, show=True):  # total_volume is like the volume of lung
    keys_list = list(id_loc_dict.keys())
    if show:
        print("we have:", len(keys_list), "number of connected components")
    id_loc_dict_sorted, id_volume_dict_sorted = sort_on_id_loc_dict(id_loc_dict)
    if total_volume is None:
        if show:
            print("the volume of these components are:\n", id_volume_dict_sorted)
    else:
        if show:
            print("total_volume is:", total_volume)
        for key in keys_list:
            if show:
                print("component", key, "constitute:", id_volume_dict_sorted[key]/total_volume, "of total volume")
    return id_loc_dict_sorted


def get_sorted_connected_regions(input_array, threshold=None, strict=False, show=True, pad=True):
    """
        :param pad:
        :param input_array: the binary mask array, with shape [x, y, z] or shape [x, y]
        :param threshold: the threshold of cast the mask array to binary
        :param strict: whether diagonal pixel is considered as adjacent.
        :param show:
        :return id_loc_dict_sorted
        """
    # key start from 1: id_loc_dict_sorted[1] is the largest; threshold > 0.5 will be considered as positive, otherwise,
    # will be considered negative

    if len(np.shape(input_array)) == 3:
        if pad:
            input_array[0, :, :] = 0
            input_array[-1, :, :] = 0
            input_array[:, 0, :] = 0
            input_array[:, -1, :] = 0
            input_array[:, :, 0] = 0
            input_array[:, :, -1] = 0

        id_loc_dict = get_connected_regions(input_array, threshold=threshold, strict=strict)[1]
        return stat_on_connected_component(id_loc_dict, show=show)
    elif len(np.shape(input_array)) == 2:
        shape = np.shape(input_array)
        temp_array = np.zeros((shape[0], shape[1], 3), 'float32')
        temp_array[:, :, 1] = input_array
        id_loc_dict = get_connected_regions(temp_array, threshold=threshold, strict=strict)[1]
        id_loc_dict_sorted = stat_on_connected_component(id_loc_dict, show=show)
        keys_list = list(id_loc_dict_sorted.keys())
        return_dict = {}
        for key in keys_list:
            return_dict[key] = list()
        for key in keys_list:
            for loc in id_loc_dict_sorted[key]:
                return_dict[key].append((loc[0], loc[1]))
        return return_dict
    else:
        raise DimensionError(input_array)


def connectedness_2d(loc_list, strict=False):
    """
    whether the loc_list forms a region that has the connectedness same to a circle?
    :param loc_list: a list of locations, like [(x1, y1), (x2, y2), ...]
    :param strict: if True, then diagonal pixel is considered as adjacent.
    :return:
    True if loc_list forms a region that has the connectedness same to a circle.
    False if otherwise, like their are more than one connected
    """
    x_min = 99999999999
    x_max = 0
    y_min = 99999999999
    y_max = 0
    for loc in loc_list:
        if loc[0] > x_max:
            x_max = loc[0]
        if loc[0] < x_min:
            x_min = loc[0]
        if loc[1] > y_max:
            y_max = loc[1]
        if loc[1] < y_min:
            y_min = loc[1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    bounding_array = np.zeros((x_range + 6, y_range + 6), 'float32')
    for loc in loc_list:
        bounding_array[loc[0] - x_min + 3, loc[1] - y_min + 3] = 1
    # we require there are only on connected component.
    assert len(list(get_sorted_connected_regions(bounding_array, strict=strict, show=False).keys())) == 1
    if not strict:
        rim_array = rim_detect.get_rim(bounding_array, outer=True)
        num_boundaries = len(list(get_sorted_connected_regions(rim_array, strict=strict, show=False).keys()))
        if num_boundaries == 1:
            return True
        else:
            print(num_boundaries)
            return False
    else:
        print("do not support strict adjacency")
        return None


def refine_connected_component(input_array, number_leave, threshold=None, strict=False, show=True, leave_min=None):
    """

    :param leave_min: float means component must have volume > leave_min * max_component_volume, int for min volume
    :param number_leave: how many connected_component to leave
    :param input_array: binary mask in 3d numpy float32
    :param threshold: if you ensure the input array is binary, set threshold to None to reduce complexity
    :param strict:
    :param show: show temporal information
    :return: array in float32 same shape with the input_array
    """
    assert len(np.shape(input_array)) == 3
    assert type(number_leave) is int
    assert number_leave > 0

    refined_array = np.zeros(np.shape(input_array), 'float32')

    loc_list_sorted = get_sorted_connected_regions(input_array, threshold, strict, show)

    if len(loc_list_sorted) == 0:
        return refined_array

    leave_count = 0
    if leave_min is not None:
        if 0 < leave_min < 1:
            min_volume = len(loc_list_sorted[1]) * leave_min
        else:
            assert leave_min > 1
            min_volume = leave_min
    else:
        min_volume = 0
    while len(loc_list_sorted) - leave_count > 0:
        if leave_count >= number_leave:
            break
        if len(loc_list_sorted[leave_count + 1]) < min_volume:
            break
        loc_array = Functions.get_location_array(loc_list_sorted[leave_count + 1])
        refined_array[loc_array] = 1
        leave_count += 1
    return refined_array


def get_connected_regions_discrete(input_array, strict=False, show=False):
    """

    :param input_array: all_file values should be int, 0 for background, non-zero for getting connected regions
    :param strict:
    :param show
    :return: a dict, {semantic_value: id_loc_dict_sorted},
    here semantic_value like 0 for background, 1 for lung, 2 for nodules
    """
    shape = np.shape(input_array)
    assert len(shape) == 2 or len(shape) == 3
    id_array = np.zeros(shape, 'int32')
    non_zero_locations_list = Functions.get_location_list(np.where(input_array > epsilon))
    non_zero_locations_list = Functions.get_location_list(np.where(input_array < -epsilon)) + non_zero_locations_list

    if show:
        print("we will check:", len(non_zero_locations_list))
        if not strict:
            estimate_time = 205 / 9530215 * len(non_zero_locations_list)
        else:
            estimate_time = 98 / 9530215 * len(non_zero_locations_list)
        if estimate_time > 3:
            print("estimate_time:", estimate_time, "s")

    id_loc_dict = {}
    id_type_dict = {}

    connected_id = 1

    for pixel_location in non_zero_locations_list:
        if id_array[pixel_location] > 0:
            # this means this point has been allocated id and volume
            continue
        else:
            locations = broadcast_connected_component_discrete(
                input_array, id_array, pixel_location, connected_id, strict=strict)
            # now, the volume and id has been broadcast to this connected component.
            id_loc_dict[connected_id] = locations
            id_type_dict[connected_id] = int(input_array[pixel_location])
            connected_id += 1  # the id is 1, 2, 3, ...

    from collections import defaultdict

    type_id_loc_dict = defaultdict(list)

    return_sorted_dict = defaultdict(dict)

    for region_id in range(1, connected_id):
        type_id_loc_dict[id_type_dict[region_id]].append((id_loc_dict[region_id], len(id_loc_dict[region_id])))

    def compare_func(tuple_a, tuple_b):
        if tuple_a[1] > tuple_b[1]:
            return 1
        return -1

    for key, value in type_id_loc_dict.items():
        value = Functions.customized_sort(value, compare_func, reverse=True)
        for type_region_id in range(len(value)):
            return_sorted_dict[key][type_region_id + 1] = value[type_region_id][0]

    return return_sorted_dict


def broadcast_connected_component_discrete(input_array, id_array, initial_location, region_id, strict):
    """

    :param input_array: the array to get the connected component
    :param id_array: the array that stores the id of connected component
    :param initial_location:
    :param region_id:
    :param strict: the definition of adjacency
    :return: the the location list of the connected component like [(389, 401), (389, 402), ..]
    """

    value_discrete = input_array[initial_location]

    shape = np.shape(input_array)

    un_labeled_region = [initial_location, ]
    id_array[initial_location] = region_id
    region_locations = []

    def visit_adjacent(location_adjacent):
        if input_array[location_adjacent] == value_discrete:
            # the adjacent pixel is in the same connected_component
            if not id_array[location_adjacent] == region_id:
                # this adjacent location is not visited
                un_labeled_region.append(location_adjacent)
                id_array[location_adjacent] = region_id  # label this unlabeled pixel

    def visit_adjacent_with_check(location_adjacent):
        if len(location_adjacent) == 3:
            if 0 <= location_adjacent[0] < shape[0] and 0 <= location_adjacent[1] < shape[1] and \
                    0 <= location_adjacent[2] < shape[2]:
                if input_array[location_adjacent] == value_discrete:
                    # the adjacent pixel is in the same connected_component
                    if not id_array[location_adjacent] == region_id:
                        # this adjacent location is not visited
                        un_labeled_region.append(location_adjacent)
                        id_array[location_adjacent] = region_id  # label this unlabeled pixel
        else:
            if 0 <= location_adjacent[0] < shape[0] and 0 <= location_adjacent[1] < shape[1]:
                if input_array[location_adjacent] == value_discrete:
                    # the adjacent pixel is in the same connected_component
                    if not id_array[location_adjacent] == region_id:
                        # this adjacent location is not visited
                        un_labeled_region.append(location_adjacent)
                        id_array[location_adjacent] = region_id  # label this unlabeled pixel

    def propagate_strict(central_loc):
        if len(central_loc) == 3:
            visit_adjacent((central_loc[0] + 1, central_loc[1], central_loc[2]))
            visit_adjacent((central_loc[0] - 1, central_loc[1], central_loc[2]))
            visit_adjacent((central_loc[0], central_loc[1] + 1, central_loc[2]))
            visit_adjacent((central_loc[0], central_loc[1] - 1, central_loc[2]))
            visit_adjacent((central_loc[0], central_loc[1], central_loc[2] + 1))
            visit_adjacent((central_loc[0], central_loc[1], central_loc[2] - 1))
        else:
            visit_adjacent((central_loc[0] + 1, central_loc[1]))
            visit_adjacent((central_loc[0] - 1, central_loc[1]))
            visit_adjacent((central_loc[0], central_loc[1] + 1))
            visit_adjacent((central_loc[0], central_loc[1] - 1))

    def propagate_strict_with_check(central_loc):
        if len(central_loc) == 3:
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1], central_loc[2]))
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1], central_loc[2]))
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1, central_loc[2]))
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1, central_loc[2]))
            visit_adjacent_with_check((central_loc[0], central_loc[1], central_loc[2] + 1))
            visit_adjacent_with_check((central_loc[0], central_loc[1], central_loc[2] - 1))
        else:
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1]))
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1]))
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1))
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1))

    def propagate_loose(central_loc):
        if len(central_loc) == 3:
            visit_adjacent((central_loc[0] - 1, central_loc[1] - 1, central_loc[2] - 1))  # (-1, -1, -1)
            visit_adjacent((central_loc[0] - 1, central_loc[1] - 1, central_loc[2]))  # (-1, -1, 0)
            visit_adjacent((central_loc[0] - 1, central_loc[1] - 1, central_loc[2] + 1))  # (-1, -1, 1)
            visit_adjacent((central_loc[0] - 1, central_loc[1], central_loc[2] - 1))  # (-1, 0, -1)
            visit_adjacent((central_loc[0] - 1, central_loc[1], central_loc[2]))  # (-1, 0, 0)
            visit_adjacent((central_loc[0] - 1, central_loc[1], central_loc[2] + 1))  # (-1, 0, 1)
            visit_adjacent((central_loc[0] - 1, central_loc[1] + 1, central_loc[2] - 1))  # (-1, 1, -1)
            visit_adjacent((central_loc[0] - 1, central_loc[1] + 1, central_loc[2]))  # (-1, 1, 0)
            visit_adjacent((central_loc[0] - 1, central_loc[1] + 1, central_loc[2] + 1))  # (-1, 1, 1)
            visit_adjacent((central_loc[0], central_loc[1] - 1, central_loc[2] - 1))  # (0, -1, -1)
            visit_adjacent((central_loc[0], central_loc[1] - 1, central_loc[2]))  # (0, -1, 0)
            visit_adjacent((central_loc[0], central_loc[1] - 1, central_loc[2] + 1))  # (0, -1, 1)
            visit_adjacent((central_loc[0], central_loc[1], central_loc[2] - 1))  # (0, 0, -1)
            visit_adjacent((central_loc[0], central_loc[1], central_loc[2] + 1))  # (0, 0, 1)
            visit_adjacent((central_loc[0], central_loc[1] + 1, central_loc[2] - 1))  # (0, 1, -1)
            visit_adjacent((central_loc[0], central_loc[1] + 1, central_loc[2]))  # (0, 1, 0)
            visit_adjacent((central_loc[0], central_loc[1] + 1, central_loc[2] + 1))  # (0, 1, 1)
            visit_adjacent((central_loc[0] + 1, central_loc[1] - 1, central_loc[2] - 1))  # (1, -1, -1)
            visit_adjacent((central_loc[0] + 1, central_loc[1] - 1, central_loc[2]))  # (1, -1, 0)
            visit_adjacent((central_loc[0] + 1, central_loc[1] - 1, central_loc[2] + 1))  # (1, -1, 1)
            visit_adjacent((central_loc[0] + 1, central_loc[1], central_loc[2] - 1))  # (1, 0, -1)
            visit_adjacent((central_loc[0] + 1, central_loc[1], central_loc[2]))  # (1, 0, 0)
            visit_adjacent((central_loc[0] + 1, central_loc[1], central_loc[2] + 1))  # (1, 0, 1)
            visit_adjacent((central_loc[0] + 1, central_loc[1] + 1, central_loc[2] - 1))  # (1, 1, -1)
            visit_adjacent((central_loc[0] + 1, central_loc[1] + 1, central_loc[2]))  # (1, 1, 0)
            visit_adjacent((central_loc[0] + 1, central_loc[1] + 1, central_loc[2] + 1))  # (1, 1, 1)
        else:
            visit_adjacent((central_loc[0] - 1, central_loc[1] - 1))  # (-1, -1)
            visit_adjacent((central_loc[0] - 1, central_loc[1]))  # (-1, 0)
            visit_adjacent((central_loc[0] - 1, central_loc[1] + 1))  # (-1, 1)
            visit_adjacent((central_loc[0], central_loc[1] - 1))  # (0, -1)
            visit_adjacent((central_loc[0], central_loc[1] + 1))  # (0, 1)
            visit_adjacent((central_loc[0] + 1, central_loc[1] - 1))  # (1, -1)
            visit_adjacent((central_loc[0] + 1, central_loc[1]))  # (1, 0)
            visit_adjacent((central_loc[0] + 1, central_loc[1] + 1))  # (1, 1)

    def propagate_loose_with_check(central_loc):
        if len(central_loc) == 3:
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] - 1, central_loc[2] - 1))  # (-1, -1, -1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] - 1, central_loc[2]))  # (-1, -1, 0)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] - 1, central_loc[2] + 1))  # (-1, -1, 1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1], central_loc[2] - 1))  # (-1, 0, -1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1], central_loc[2]))  # (-1, 0, 0)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1], central_loc[2] + 1))  # (-1, 0, 1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] + 1, central_loc[2] - 1))  # (-1, 1, -1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] + 1, central_loc[2]))  # (-1, 1, 0)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] + 1, central_loc[2] + 1))  # (-1, 1, 1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1, central_loc[2] - 1))  # (0, -1, -1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1, central_loc[2]))  # (0, -1, 0)
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1, central_loc[2] + 1))  # (0, -1, 1)
            visit_adjacent_with_check((central_loc[0], central_loc[1], central_loc[2] - 1))  # (0, 0, -1)
            visit_adjacent_with_check((central_loc[0], central_loc[1], central_loc[2] + 1))  # (0, 0, 1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1, central_loc[2] - 1))  # (0, 1, -1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1, central_loc[2]))  # (0, 1, 0)
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1, central_loc[2] + 1))  # (0, 1, 1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] - 1, central_loc[2] - 1))  # (1, -1, -1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] - 1, central_loc[2]))  # (1, -1, 0)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] - 1, central_loc[2] + 1))  # (1, -1, 1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1], central_loc[2] - 1))  # (1, 0, -1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1], central_loc[2]))  # (1, 0, 0)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1], central_loc[2] + 1))  # (1, 0, 1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] + 1, central_loc[2] - 1))  # (1, 1, -1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] + 1, central_loc[2]))  # (1, 1, 0)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] + 1, central_loc[2] + 1))  # (1, 1, 1)
        else:
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] - 1))  # (-1, -1)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1]))  # (-1, 0)
            visit_adjacent_with_check((central_loc[0] - 1, central_loc[1] + 1))  # (-1, 1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] - 1))  # (0, -1)
            visit_adjacent_with_check((central_loc[0], central_loc[1] + 1))  # (0, 1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] - 1))  # (1, -1)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1]))  # (1, 0)
            visit_adjacent_with_check((central_loc[0] + 1, central_loc[1] + 1))  # (1, 1)

    if strict:
        while un_labeled_region:  # this mean un_labeled_region is not empty
            location = un_labeled_region.pop()

            region_locations.append(location)  # get the locations of the connected component

            if len(location) == 3:
                if 0 < location[0] < (shape[0] - 1) and 0 < location[1] < (shape[1] - 1) and \
                        0 < location[2] < (shape[2] - 1):
                    propagate_strict(location)
                else:
                    propagate_strict_with_check(location)
            else:
                if 0 < location[0] < (shape[0] - 1) and 0 < location[1] < (shape[1] - 1):
                    propagate_strict(location)
                else:
                    propagate_strict_with_check(location)
    else:
        while un_labeled_region:  # this mean un_labeled_region is not empty
            location = un_labeled_region.pop()

            region_locations.append(location)  # get the locations of the connected component

            if len(location) == 3:
                if 0 < location[0] < (shape[0] - 1) and 0 < location[1] < (shape[1] - 1) and \
                        0 < location[2] < (shape[2] - 1):
                    propagate_loose(location)
                else:
                    propagate_loose_with_check(location)
            else:
                if 0 < location[0] < (shape[0] - 1) and 0 < location[1] < (shape[1] - 1):
                    propagate_loose(location)
                else:
                    propagate_loose_with_check(location)

    return region_locations


def convert_to_simply_connected_old(stack_region_mask, dimension=2, parallel_count=None, iter_round=2,
                                    add_outer_layer=0,
                                    return_array_dtype='float16'):
    """

    :param add_outer_layer:
    :param iter_round:
    :param stack_region_mask: numpy array in shape [batch, x, y] or [x, y] or [batch, x, y, z] or [x, y, z]

        for each region_mask, there should be only one connect region

    :param dimension: the dimension of the region mask, 2 or 3
    :param parallel_count: None for not parallel, int for max parallel count. The parallel is on batch level
    :param return_array_dtype:
    :return: stack_region_mask that are simply connected
    """

    if iter_round == 0:
        return stack_region_mask

    assert dimension == 2 or dimension == 3
    shape_stack = np.shape(stack_region_mask)
    assert len(shape_stack) - 1 <= dimension <= len(shape_stack)

    for layer in range(add_outer_layer):
        if dimension == 2:
            rim_or_surface = get_surface_and_rim.get_rim(stack_region_mask, outer=True, strict=False)
        else:
            rim_or_surface = get_surface_and_rim.get_surface(stack_region_mask, outer=True, strict=False)
        # extend surface to remove holes
        stack_region_mask = stack_region_mask + rim_or_surface

    def trim_boundary_to_zero(array):
        shape_array = np.shape(array)
        if len(shape_array) == 2:
            array[0, :] = 0
            array[shape_array[0] - 1, :] = 0
            array[:, 0] = 0
            array[:, shape_array[1] - 1] = 0
        if len(shape_array) == 3:
            array[0, :, :] = 0
            array[shape_array[0] - 1, :, :] = 0
            array[:, 0, :] = 0
            array[:, shape_array[1] - 1, :] = 0
            array[:, :, 0] = 0
            array[:, :, shape_array[2] - 1] = 0

    if len(shape_stack) > dimension:
        for slice_id in range(shape_stack[0]):
            trim_boundary_to_zero(stack_region_mask[slice_id])
    else:
        trim_boundary_to_zero(stack_region_mask)

    if dimension == 2:
        rim_or_surface = get_surface_and_rim.get_rim(stack_region_mask, outer=True, strict=False)
    else:
        rim_or_surface = get_surface_and_rim.get_surface(stack_region_mask, outer=True, strict=False)

    return_array = np.array(stack_region_mask, return_array_dtype)

    if len(shape_stack) > dimension and parallel_count is not None:
        input_list = []
        for slice_id in range(shape_stack[0]):
            input_list.append((rim_or_surface[slice_id], return_array[slice_id]))
        output_list = Functions.func_parallel(
            derive_topological_connectivity_mask, input_list, parallel_count=parallel_count)
        for slice_id in range(shape_stack[0]):
            return_array[slice_id] = output_list[slice_id]
    else:
        if len(shape_stack) == dimension:
            return_array = derive_topological_connectivity_mask((rim_or_surface, return_array))
        else:
            for slice_id in range(shape_stack[0]):
                return_array[slice_id] = \
                    derive_topological_connectivity_mask((rim_or_surface[slice_id], return_array[slice_id]))

    return convert_to_simply_connected_old(return_array, dimension, parallel_count, iter_round - 1, 0)


def derive_topological_connectivity_mask(input_tuple):

    rim_or_surface_mask, original_image = input_tuple
    type_sorted_loc_dict = get_connected_regions_discrete(rim_or_surface_mask, strict=True)
    sorted_loc_dict = type_sorted_loc_dict[1]
    num_regions = len(sorted_loc_dict)
    for inside_region_id in range(2, num_regions + 1):
        location_list = sorted_loc_dict[inside_region_id]
        if len(location_list) > 100:
            original_image[Functions.get_location_array(location_list)] = 1
        else:
            for location in location_list:
                original_image[location] = 1
    return original_image


def convert_to_simply_connected(region_mask, max_hole_radius=1, add_surface=0):
    """

    the region mask must only contain one connected component
    this function removes holes

    :param add_surface: add extra surface to the original mask
    :param max_hole_radius:
    :param region_mask: numpy array in shape [x, y] or [x, y, z]

        for each region_mask, there should be only one connect region

    :return: stack_region_mask that are simply connected
    """
    dimension = len(np.shape(region_mask))
    assert dimension == 2 or dimension == 3

    for layer in range(max_hole_radius):
        if dimension == 2:
            rim_or_surface = get_surface_and_rim.get_rim(region_mask, outer=True, strict=False)
        else:
            rim_or_surface = get_surface_and_rim.get_surface(region_mask, outer=True, strict=False)
        # extend surface to remove holes
        region_mask = region_mask + rim_or_surface

        if add_surface < 1:
            # remove the outer surface
            outer_surface_loc_list = get_sorted_connected_regions(
                rim_or_surface, threshold=None, strict=False, show=False)[1]
            region_mask[Functions.get_location_array(outer_surface_loc_list)] = 0
        else:
            add_surface -= 1

    while add_surface > 0:
        if dimension == 2:
            rim_or_surface = get_surface_and_rim.get_rim(region_mask, outer=True, strict=False)
        else:
            rim_or_surface = get_surface_and_rim.get_surface(region_mask, outer=True, strict=False)
        # extend surface to remove holes
        region_mask = region_mask + rim_or_surface
        add_surface -= 1

    return region_mask


def propagate_to_wider_region(valid_array, seed_region_array, strict=False, return_id_loc_dict=False):
    """

    if there are overlap between valid_array and seed_region_array, propagate overlap region to fill valid_array

    :param valid_array: the mask array, with shape [x, y, z]
    :param seed_region_array: the mask array, with shape [x, y, z]
    :param strict: whether diagonal pixel is considered as adjacent.

    :return: refined valid_array, or id_loc_dict_sorted for refined_valid_array
    """

    shape = np.shape(valid_array)
    helper_array = np.zeros([shape[0], shape[1], shape[2], 2])
    # the last dim has two channels, the first is the key, the second is the volume
    helper_array[:, :, :, 0] = -valid_array

    overlap_region_mask = valid_array * seed_region_array
    overlap_region = np.where(overlap_region_mask > 0)

    num_checking_points = len(overlap_region[0])
    # print("we will check:", num_checking_points)
    id_volume_dict = {}
    id_loc_dict = {}

    connected_id = 1

    for index in range(num_checking_points):
        pixel_location = (overlap_region[0][index], overlap_region[1][index], overlap_region[2][index])
        if helper_array[pixel_location[0], pixel_location[1], pixel_location[2], 0] > epsilon:
            # this means this point has been allocated id and volume
            continue
        else:
            # this means this point has been allocated id and volume
            if strict:
                volume, locations = broadcast_connected_component(helper_array, pixel_location, connected_id)
            else:
                volume, locations = broadcast_connected_component_2(helper_array, pixel_location, connected_id)
            # now, the volume and id has been broadcast to this connected component.
            id_volume_dict[connected_id] = volume
            id_loc_dict[connected_id] = locations
            connected_id += 1  # the id is 1, 2, 3, ...

    if not return_id_loc_dict:
        return np.array(helper_array[:, :, :, 1] > 0, 'float32')

    return sort_on_id_loc_dict(id_loc_dict, id_volume_dict)[0]


if __name__ == '__main__':

    exit()
