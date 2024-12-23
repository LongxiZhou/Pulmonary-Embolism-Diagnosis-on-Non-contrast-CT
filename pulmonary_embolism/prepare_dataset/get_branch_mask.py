"""

input the depth_array and center line mask,
output the branch mask. root: branch 0

branch is based on the encoding_depth of the vessel

"""
import numpy as np
import Tool_Functions.Functions as Functions
import collections
import math


def get_depth_for_center_line_point(center_location, depth_array, radius=5, convert_to_branch=True):
    """

    :param center_location: (x, y, z)
    :param radius:
    :param depth_array
    :param convert_to_branch
    :return: the encoding_depth, or branch for the center line location
    """

    x, y, z = center_location

    xy_array = depth_array[x - radius: x + radius, y - radius: y + radius, z]
    yz_array = depth_array[x, y - radius: y + radius, z - radius: z + radius]
    xz_array = depth_array[x - radius: x + radius, y, z - radius: z + radius]

    depth_for_location = max(np.max(xy_array), np.max(yz_array), np.max(xz_array))

    if not convert_to_branch:
        return depth_for_location
    return math.log(depth_for_location / 30) / math.log(0.7)


def smooth_depth(center_location, center_line_array_depth, protocol_func, radius=1):
    """

    :param protocol_func: protocol_func(flatten_cube, non_zero_count)
    :param center_location:
    :param center_line_array_depth:
    :param radius:
    :return:
    """
    x, y, z = center_location
    search_cube = center_line_array_depth[x - radius: x + radius, y - radius: y + radius, z - radius: z + radius]

    non_zero_count = len(np.where(search_cube > 0.5)[0])

    flatten_cube = -np.reshape(search_cube, (-1, ))

    flatten_cube = -np.sort(flatten_cube)

    if non_zero_count == 0:
        return 0

    return protocol_func(flatten_cube, non_zero_count)


def get_branch_for_center_line(center_line_mask, depth_array, radius=5, radius_smooth=1, interval=1):
    """

    :param interval:
    :param center_line_mask:
    :param depth_array:
    :param radius:
    :param radius_smooth
    :return: center line array with brach
    """
    assert type(interval) is int and interval >= 1
    assert radius_smooth >= interval  # otherwise smooth has no effect

    shape = np.shape(center_line_mask)

    loc_list_center_line_temp = Functions.get_location_list(np.where(center_line_mask > 0.5))

    loc_list_center_line = []
    for loc in loc_list_center_line_temp:
        if depth_array[loc] >= 1:
            loc_list_center_line.append(loc)

    print("there are", len(loc_list_center_line), "center line points")
    # assert len(loc_list_center_line_temp) - len(loc_list_center_line) == 0

    if interval > 1:
        new_loc_list = []
        for location in loc_list_center_line:
            if location[0] % interval == 0 and location[1] % interval == 0 and location[2] % interval == 0:
                new_loc_list.append(location)

        loc_list_center_line = new_loc_list

    temp_array = np.zeros(np.shape(center_line_mask), 'float16')

    for location in loc_list_center_line:
        if not (min(location) > radius and
                min(shape[0] - location[0], shape[1] - location[1], shape[2] - location[2]) > radius + 1):
            continue

        temp_array[location] = get_depth_for_center_line_point(location, depth_array, radius, convert_to_branch=True)

    if radius_smooth is None:
        return temp_array

    return_array = np.zeros(np.shape(center_line_mask), 'float16')

    def protocol_func(flatten_cube, non_zero_count):
        return np.sum(flatten_cube[0: non_zero_count]) / non_zero_count

    for location in loc_list_center_line:
        assert min(location) > radius_smooth and \
               min(shape[0] - location[0], shape[1] - location[1], shape[2] - location[2]) > radius_smooth + 1

        return_array[location] = smooth_depth(location, temp_array, protocol_func, radius_smooth)

    return return_array


def propagate_branching_cloud(center_line_depth_array, blood_vessel_mask, step=2, weight_half_decay=25):
    """

    :param center_line_depth_array:
    :param blood_vessel_mask:
    :param step:
    :param weight_half_decay:
    :return: a point cloud same shape with blood_vessel_mask,
    non-zero is the max encoding_depth of the nearest center line point.
    """

    assert type(step) is int and step >= 1
    assert weight_half_decay is None or weight_half_decay > 0
    if weight_half_decay is None:
        use_weight = False
        weight_half_decay = 10
    else:
        use_weight = True

    value_dict = collections.defaultdict(list)

    weight_dict = collections.defaultdict(list)

    point_cloud_array = np.zeros(np.shape(blood_vessel_mask), 'float16')

    loc_list_center_line = Functions.get_location_list(np.where(center_line_depth_array > 0.5))

    if step > 1:
        new_loc_list = []
        for location in loc_list_center_line:
            if location[0] % step == 0 and location[1] % step == 0 and location[2] % step == 0:
                new_loc_list.append(location)

        loc_list_center_line = new_loc_list

    def distance(pointer):
        return math.sqrt((pointer[0] - origin_location[0]) ** 2 + (pointer[1] - origin_location[1]) ** 2 +
                         (pointer[2] - origin_location[2]) ** 2)

    def broad_cast_depth_1d(initial_loc, depth, axis, broad_cast_initial=True):
        """

        :param initial_loc: [x, y, z]
        :param depth: the encoding_depth to be broadcast
        :param axis: int, 0 for 'x', 1 for 'y', 2 for 'z'
        :param broad_cast_initial: whether the initial loc be assign encoding_depth
        :return: None
        """
        pointer = list(initial_loc)
        if not broad_cast_initial:
            pointer[axis] += step
        pointer_tuple = tuple(pointer)
        while blood_vessel_mask[pointer_tuple] > 0.5:
            value_dict[pointer_tuple].append(depth)
            weight_dict[pointer_tuple].append(2 ** -distance(pointer_tuple) / weight_half_decay)
            pointer[axis] += step
            pointer_tuple = tuple(pointer)

        pointer = list(initial_loc)
        pointer[axis] -= step
        pointer_tuple = tuple(pointer)
        while blood_vessel_mask[pointer_tuple] > 0.5:
            value_dict[pointer_tuple].append(depth)
            weight_dict[pointer_tuple].append(2 ** -distance(pointer_tuple) / weight_half_decay)
            pointer[axis] -= step
            pointer_tuple = tuple(pointer)

    def broad_cast_depth_2d(initial_loc, depth, axis_vertical):
        """

        :param initial_loc: (x, y, z)
        :param depth:
        :param axis_vertical: the axis that vertical to the 2d plane to broad cast
        :return: None
        """
        pointer = list(initial_loc)

        axis_broad_cast = (axis_vertical + 1) % 3
        axis_move = (axis_vertical + 2) % 3

        broad_cast_depth_1d(pointer, depth, axis_broad_cast, False)

        pointer[axis_move] += step
        while blood_vessel_mask[tuple(pointer)] > 0.5:
            broad_cast_depth_1d(pointer, depth, axis_broad_cast, True)
            pointer[axis_move] += step

        pointer = list(initial_loc)
        pointer[axis_move] -= step
        while blood_vessel_mask[tuple(pointer)] > 0.5:
            broad_cast_depth_1d(pointer, depth, axis_broad_cast, True)
            pointer[axis_move] -= step

    def broad_cast_coordinate(origin_loc, depth):
        broad_cast_depth_2d(origin_loc, depth, 0)
        broad_cast_depth_2d(origin_loc, depth, 1)
        broad_cast_depth_2d(origin_loc, depth, 2)

    def weighted_average(list_value, list_weight):
        weight_sum = np.sum(list_weight)
        return_value = 0.
        for index, value in enumerate(list_value):
            return_value += value * list_weight[index] / weight_sum
        return return_value

    for location in loc_list_center_line:

        depth_value = center_line_depth_array[location]
        origin_location = location
        broad_cast_coordinate(origin_location, depth_value)

    if use_weight:

        for location, value_list in value_dict.items():

            weight_list = weight_dict[location]

            averaged_branch = weighted_average(value_list, weight_list)

            point_cloud_array[location] = averaged_branch

    else:
        for location, value_list in value_dict.items():

            averaged_branch = np.average(value_list)

            point_cloud_array[location] = averaged_branch

    return point_cloud_array


def get_branching_cloud(center_line_mask, depth_array, search_radius=5, smooth_radius=1, step=2,
                        weight_half_decay=20, refine_radius=4):
    center_line_branch_array = get_branch_for_center_line(
        center_line_mask, depth_array, search_radius, smooth_radius, 1)

    branching_cloud = propagate_branching_cloud(center_line_branch_array, depth_array, step, weight_half_decay)

    if refine_radius > 0:
        location_list = Functions.get_location_list(np.where(branching_cloud > 0))

        refined_branching_cloud = np.zeros(np.shape(branching_cloud), 'float16')

        def protocol_func(flatten_cube, non_zero_count):
            # get the min value from the non-zeros
            return flatten_cube[non_zero_count - 1]

        for location in location_list:
            refined_branching_cloud[location] = smooth_depth(location, branching_cloud, protocol_func, refine_radius)

        return refined_branching_cloud

    return branching_cloud


if __name__ == '__main__':

    import os
    import analysis.get_surface_rim_adjacent_mean as get_surface
    import visualization.visualize_3d.visualize_stl as stl

    test_center_line = np.load(
        '/data_disk/artery_vein_project/new_data/CTA/depth_and_center-line/blood_center_line/AL00004.npz')['array']
    test_depth_array = np.load(
        '/data_disk/artery_vein_project/new_data/CTA/depth_and_center-line/depth_array/AL00004.npz')['array']
    test_branching_cloud = get_branching_cloud(test_center_line, test_depth_array, step=1)
    for i in range(200, 300, 3):
        Functions.image_show(test_branching_cloud[:, :, i])

    exit()

    fn_list = os.listdir('/data_disk/RAD-ChestCT_dataset/depth_and_center-line/depth_array/')

    report_dict = Functions.pickle_load_object('/data_disk/RAD-ChestCT_dataset/report_dict.pickle')

    fn_name = fn_list[3]

    blood_center_line = \
    np.load('/data_disk/RAD-ChestCT_dataset/depth_and_center-line/blood_center_line/' + fn_name)['array']
    blood_depth_mask = np.load('/data_disk/RAD-ChestCT_dataset/depth_and_center-line/depth_array/' + fn_name)[
        'array']

    print(report_dict[fn_name[:-4]])

    stl.visualize_numpy_as_stl(np.array(blood_depth_mask > 0.5, 'float32'))

    surface = get_surface.get_surface(np.array(blood_depth_mask > 0.5, 'float32'), strict=True, outer=False)

    test_branching_cloud = get_branching_cloud(blood_center_line, blood_depth_mask, step=1)

    print("max branching", np.max(test_branching_cloud))

    for i in range(250, 300, 3):
        image_left = test_branching_cloud[:, :, i]

        max_branch = np.max(test_branching_cloud)

        image_right = surface[:, :, i] * max_branch

        image = np.concatenate((image_left, image_right), axis=1)
        Functions.image_show(image)

    exit()
