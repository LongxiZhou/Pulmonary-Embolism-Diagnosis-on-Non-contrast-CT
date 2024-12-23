"""
input: center-line array in binary, numpy array shaped like [x, y, z]
output:
"""

import numpy as np
import analysis.connect_region_detect as connect_region_detect
import Tool_Functions.Functions as Functions


def annotate_level_for_center_line(center_line_array):
    """

    :param center_line_array: binary, numpy array shaped like [x, y, z], should only contains one connected component
    :return: data structure: {connected_component_id: {branching_level: {stick_id: [(a, b, c), ...], ...}, ...}, ...}
    branching_level -1 means circle locations
    """
    shape_array = np.shape(center_line_array)
    assert len(shape_array) == 3
    id_loc_dict = connect_region_detect.get_sorted_connected_regions(center_line_array)

    return_data = {}

    for connected_component_id, loc_list in id_loc_dict.items():

        level_stick_loc_dict = get_level_stick_dict(set(loc_list))

        return_data[connected_component_id] = level_stick_loc_dict

    return return_data


def get_level_stick_dict(loc_set):
    """

    :param loc_set: {(x, y, z), ...}. These locations forms a connected component
    :return: {branching_level: {stick_id: [(a, b, c), ...], ...}, ...}
    """

    level_stick_loc_dict = {}
    branching_level = 0

    def annotate_one_level():

        stick_id = 0
        list_loc_tail = get_tail_voxels(loc_set)

        if len(list_loc_tail) == 0:
            return True

        stick_id_loc_dict = {}
        for tail_loc in list_loc_tail:

            stick_loc_list = get_stick_loc_list_and_remove_stick_from_loc_set(tail_loc, loc_set)
            stick_id_loc_dict[stick_id] = stick_loc_list
            stick_id += 1

        level_stick_loc_dict[branching_level] = stick_id_loc_dict

        return False

    while len(loc_set) > 0:
        only_circle_remaining = annotate_one_level()
        branching_level += 1
        if only_circle_remaining:
            level_stick_loc_dict[-1] = list(loc_set)
            break

    return level_stick_loc_dict


def get_stick_loc_list_and_remove_stick_from_loc_set(tail_loc, loc_set):

    def get_adjacent_locations(loc):
        x, y, z = loc
        adjacent_location_set = \
            {(x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x + 1, y + 1, z - 1),
             (x + 1, y, z + 1), (x + 1, y, z), (x + 1, y, z - 1),
             (x + 1, y - 1, z + 1), (x + 1, y - 1, z), (x + 1, y - 1, z - 1),

             (x, y + 1, z + 1), (x, y + 1, z), (x, y + 1, z - 1),
             (x, y, z + 1), (x, y, z - 1),
             (x, y - 1, z + 1), (x, y - 1, z), (x, y - 1, z - 1),

             (x - 1, y + 1, z + 1), (x - 1, y + 1, z), (x - 1, y + 1, z - 1),
             (x - 1, y, z + 1), (x - 1, y, z), (x - 1, y, z - 1),
             (x - 1, y - 1, z + 1), (x - 1, y - 1, z), (x - 1, y - 1, z - 1)
             }

        intersect = adjacent_location_set & loc_set

        return intersect

    stick_loc_list = []

    adjacent_set = get_adjacent_locations(tail_loc)
    stick_loc_list.append(tail_loc)
    loc_set.difference_update(set(tail_loc))

    while len(adjacent_set) == 1:
        stick_loc_list.append(tail_loc)
        loc_set.difference_update(adjacent_set)  # remove old one
        tail_loc = adjacent_set.pop()  # the new tail
        adjacent_set = get_adjacent_locations(tail_loc)

    for remaining_loc in adjacent_set:
        loc_set.remove(remaining_loc)
        stick_loc_list.append(remaining_loc)

    return stick_loc_list


def get_tail_voxels(loc_set):
    """

    :param loc_set: {(x, y, z), ...}. These locations forms a connected component
    :return: list, item is (x, y, z, adjacent_count)
    """

    def get_adjacent_count(loc):
        x, y, z = loc
        adjacent_location_set = \
            {(x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x + 1, y + 1, z - 1),
             (x + 1, y, z + 1), (x + 1, y, z), (x + 1, y, z - 1),
             (x + 1, y - 1, z + 1), (x + 1, y - 1, z), (x + 1, y - 1, z - 1),

             (x, y + 1, z + 1), (x, y + 1, z), (x, y + 1, z - 1),
             (x, y, z + 1), (x, y, z - 1),
             (x, y - 1, z + 1), (x, y - 1, z), (x, y - 1, z - 1),

             (x - 1, y + 1, z + 1), (x - 1, y + 1, z), (x - 1, y + 1, z - 1),
             (x - 1, y, z + 1), (x - 1, y, z), (x - 1, y, z - 1),
             (x - 1, y - 1, z + 1), (x - 1, y - 1, z), (x - 1, y - 1, z - 1)
             }

        intersect = adjacent_location_set & loc_set

        return len(intersect)

    list_loc_tail = []

    for location in loc_set:
        adjacent_count = get_adjacent_count(location)
        if adjacent_count == 1:
            list_loc_tail.append(location)

    return list_loc_tail


if __name__ == '__main__':
    test_array = np.load('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/depth_and_center-line/blood_center_line/patient-id-20858924.npz')['array']

    import time

    loc_list_test = Functions.get_location_list(np.where(test_array > 0.5))

    loc_set_test = set(loc_list_test)

    start_time = time.time()

    data_test = annotate_level_for_center_line(test_array)

    end_time = time.time()

    lest = data_test[1][-1][0]

    Functions.show_point_cloud_3d(lest, data_type='list')

    print("time cost:", end_time - start_time)



