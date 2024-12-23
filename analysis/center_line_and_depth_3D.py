import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.connect_region_detect as connectivity
import Tool_Functions.Functions as Functions
import skimage.morphology as morphology
import numpy as np


search_range = 0

loc_list_layer_one_voxel = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2),
                            (1, 3, 3), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 3), (2, 3, 1), (2, 3, 2),
                            (2, 3, 3), (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 1),
                            (3, 3, 2), (3, 3, 3)]
temp_array_for_central_locations = np.ones([5, 5, 5], 'float32')
temp_array_for_central_locations[1: 4, 1: 4, 1: 4] = 0
loc_list_layer_two_voxel = Functions.get_location_list(np.where(temp_array_for_central_locations > 0.5))

# should remove (1, 1, 1)
adjacent_list_000 = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]

adjacent_list_001 = [(0, 0, 0), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
                     (1, 1, 2)]
adjacent_list_002 = [(0, 0, 1), (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 0, 2), (1, 1, 2)]

adjacent_list_010 = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
                     (1, 2, 1)]
adjacent_list_011 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0),
                     (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2)]
adjacent_list_012 = [(0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 2, 1), (0, 2, 2), (1, 0, 1), (1, 0, 2), (1, 1, 2), (1, 2, 1),
                     (1, 2, 2)]
adjacent_list_020 = [(0, 1, 0), (0, 1, 1), (0, 2, 1), (1, 1, 0), (1, 2, 0), (1, 2, 1)]

adjacent_list_021 = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 0), (1, 2, 1),
                     (1, 2, 2)]
adjacent_list_022 = [(0, 1, 1), (0, 1, 2), (0, 2, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2)]

adjacent_list_100 = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0),
                     (2, 1, 1)]
adjacent_list_101 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 2), (1, 1, 0),
                     (1, 1, 2), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2)]
adjacent_list_102 = [(0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2), (2, 0, 1), (2, 0, 2), (2, 1, 1),
                     (2, 1, 2)]
adjacent_list_110 = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 2, 0),
                     (1, 2, 1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)]
adjacent_list_112 = [(0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (1, 0, 1), (1, 0, 2), (1, 2, 1),
                     (1, 2, 2), (2, 0, 1), (2, 0, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
adjacent_list_120 = [(0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 1, 0), (1, 2, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0),
                     (2, 2, 1)]
adjacent_list_121 = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 0),
                     (1, 2, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)]
adjacent_list_122 = [(0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 1, 2), (2, 2, 1),
                     (2, 2, 2)]
adjacent_list_200 = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]

adjacent_list_201 = [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 2), (2, 0, 0), (2, 0, 2), (2, 1, 0), (2, 1, 1),
                     (2, 1, 2)]
adjacent_list_202 = [(1, 0, 1), (1, 0, 2), (1, 1, 2), (2, 0, 1), (2, 1, 1), (2, 1, 2)]

adjacent_list_210 = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0), (1, 2, 1), (2, 0, 0), (2, 0, 1), (2, 1, 1), (2, 2, 0),
                     (2, 2, 1)]
adjacent_list_211 = [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0),
                     (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)]
adjacent_list_212 = [(1, 0, 1), (1, 0, 2), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 0, 1), (2, 0, 2), (2, 1, 1), (2, 2, 1),
                     (2, 2, 2)]
adjacent_list_220 = [(1, 1, 0), (1, 2, 0), (1, 2, 1), (2, 1, 0), (2, 1, 1), (2, 2, 1)]

adjacent_list_221 = [(1, 1, 0), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0),
                     (2, 2, 2)]
adjacent_list_222 = [(1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1)]


def get_surface_distance(binary_mask, threshold=None, strict=True, return_max_distance=False):
    """
    calculate the distance to nearest surface for every positive voxel
    :param return_max_distance: further return the max_distance
    :param binary_mask: should in shape [a, b, c]
    :param threshold: if not None, apply binary_mask = np.array(binary_mask > threshold, 'float32')
    :param strict: distance is calculated by how many times strip the surface to reach a voxel.
    strict is True: adjacent voxel is 6 (strict=False: adjacent voxel is 26)
    :return: array shape shape with input binary_mask, each positive voxel is the times striping the surface to reach.
    """
    if threshold is None:
        temp_array = np.array(binary_mask > 0, 'float32')  # temp_array is to protect the binary_mask
    else:
        temp_array = np.array(binary_mask > threshold, 'float32')

    # pad two into zero
    temp_array[0:2, :, :] = 0
    temp_array[-1, :, :] = 0
    temp_array[-2, :, :] = 0
    temp_array[:, 0:2, :] = 0
    temp_array[:, -1, :] = 0
    temp_array[:, -2, :] = 0
    temp_array[:, :, 0:2] = 0
    temp_array[:, :, -1] = 0
    temp_array[:, :, -2] = 0

    if not np.sum(temp_array) > 0:
        print("no data input")
        return np.zeros(np.shape(temp_array), 'float32')

    return_array = np.array(temp_array, 'float32')

    bounding_box_list = [Functions.get_bounding_box(temp_array, pad=1)]
    # no need to calculate surface on the entire volume. Will accelerate more than 10 times.

    def add_to_original(remaining_component):
        """
        add the remaining_component to the return_array during striping the surface
        :param remaining_component: binary array
        :return: None
        """
        start_x, _ = bounding_box_list[0][0]
        start_y, _ = bounding_box_list[0][1]
        start_z, _ = bounding_box_list[0][2]
        for shift in bounding_box_list[1::]:
            start_x += shift[0][0]
            start_y += shift[1][0]
            start_z += shift[2][0]
        end_x = start_x + bounding_box_list[-1][0][1] - bounding_box_list[-1][0][0]
        end_y = start_y + bounding_box_list[-1][1][1] - bounding_box_list[-1][1][0]
        end_z = start_z + bounding_box_list[-1][2][1] - bounding_box_list[-1][2][0]

        return_array[start_x: end_x, start_y: end_y, start_z: end_z] = \
            return_array[start_x: end_x, start_y: end_y, start_z: end_z] + remaining_component

    temp_array = temp_array[bounding_box_list[-1][0][0]: bounding_box_list[-1][0][1],
                            bounding_box_list[-1][1][0]: bounding_box_list[-1][1][1],
                            bounding_box_list[-1][2][0]: bounding_box_list[-1][2][1]]

    count = 0
    while True:
        count += 1
        surface = get_surface.get_surface(temp_array, strict=strict)
        temp_array = temp_array - surface
        add_to_original(temp_array)
        if not np.sum(temp_array) > 0:
            break
        bounding_box_list.append(Functions.get_bounding_box(temp_array, pad=1))
        temp_array = temp_array[bounding_box_list[-1][0][0]: bounding_box_list[-1][0][1],
                                bounding_box_list[-1][1][0]: bounding_box_list[-1][1][1],
                                bounding_box_list[-1][2][0]: bounding_box_list[-1][2][1]]

    if return_max_distance:
        return return_array, count
    return return_array


def stratify_surface_distance(binary_mask, threshold=None, strict=True):
    """
    calculate the distance to nearest surface for every positive voxel
    :param binary_mask: should in shape [a, b, c]
    :param threshold: if not None, apply binary_mask = np.array(binary_mask > threshold, 'float32')
    :param strict: distance is calculated by how many times strip the surface to reach a voxel.
    strict is True: adjacent voxel is 6 (strict=False: adjacent voxel is 26)
    :return: dict of location list, key is the surface distance (int) start from 1.  {1: [(x, y, z), ...], ...}
    """
    if threshold is None:
        temp_array = np.array(binary_mask, 'float32')  # temp_array is to protect the binary_mask
    else:
        temp_array = np.array(binary_mask > threshold, 'float32')

    return_dict = {}

    assert np.sum(temp_array) > 0

    bounding_box_list = [Functions.get_bounding_box(temp_array, pad=1)]
    # no need to calculate surface on the entire volume. Will accelerate more than 10 times.
    temp_array = temp_array[bounding_box_list[-1][0][0]: bounding_box_list[-1][0][1],
                            bounding_box_list[-1][1][0]: bounding_box_list[-1][1][1],
                            bounding_box_list[-1][2][0]: bounding_box_list[-1][2][1]]

    def add_to_return_dict(current_surface, current_distance):
        """
        add the remaining_component to the return_array during striping the surface
        :param current_distance: the distance for current_surface to the outer space
        :param current_surface: the surface mask (tight)
        :return: None
        """
        start_x, _ = bounding_box_list[0][0]
        start_y, _ = bounding_box_list[0][1]
        start_z, _ = bounding_box_list[0][2]
        for shift in bounding_box_list[1::]:
            start_x += shift[0][0]
            start_y += shift[1][0]
            start_z += shift[2][0]

        loc_array_current_surface = list(np.where(current_surface == 1))
        loc_array_current_surface[0] = loc_array_current_surface[0] + start_x
        loc_array_current_surface[1] = loc_array_current_surface[1] + start_y
        loc_array_current_surface[2] = loc_array_current_surface[2] + start_z

        return_dict[current_distance] = Functions.get_location_list(loc_array_current_surface)

    count = 0
    while True:
        # print(count)
        count += 1
        surface = get_surface.get_surface(temp_array, strict=strict)
        add_to_return_dict(surface, count)
        temp_array = temp_array - surface
        if not np.sum(temp_array) > 0:
            break
        bounding_box_list.append(Functions.get_bounding_box(temp_array, pad=1))
        temp_array = temp_array[bounding_box_list[-1][0][0]: bounding_box_list[-1][0][1],
                                bounding_box_list[-1][1][0]: bounding_box_list[-1][1][1],
                                bounding_box_list[-1][2][0]: bounding_box_list[-1][2][1]]

    return return_dict


def abstract_center_line_3d(binary_mask, threshold=None):
    """
    each cube (64, 64, 64), use the center (48, 48, 48)
    :param binary_mask: should in shape [a, b, c]
    :param threshold: if not None, apply binary_mask = np.array(binary_mask > threshold, 'float32')
    :return: binary array for center line, 1 means center line, 0 means not center line
    """
    if threshold is not None:
        binary_mask = np.array(binary_mask > threshold, 'float32')
    center_line_array = np.zeros(np.shape(binary_mask), 'float32')

    cube_list = cube_slicer(binary_mask, (64, 64, 64), (48, 48, 48))

    non_zero_cube_list = []

    for item in cube_list:
        if np.sum(item[1]) > 0:
            non_zero_cube_list.append(item)

    center_line_cube_list = Functions.func_parallel(basic_center_line_abstract, non_zero_cube_list)

    for item in center_line_cube_list:
        location = item[0]
        cube_center_line_array = item[1]

        center_line_array[location[0] + 8: location[0] + 56, location[1] + 8: location[1] + 56,
                          location[2] + 8: location[2] + 56] = cube_center_line_array[8: 56, 8: 56, 8: 56]

    return center_line_array


def abstract_center_line_3d_faster(binary_mask, threshold=None):
    """
    each cube (32, 32, 32), use the center (28, 28, 28)
    :param binary_mask: should in shape [a, b, c]
    :param threshold: if not None, apply binary_mask = np.array(binary_mask > threshold, 'float32')
    :return: binary array for center line, 1 means center line, 0 means not center line
    """
    if threshold is not None:
        binary_mask = np.array(binary_mask > threshold, 'float32')
    center_line_array = np.zeros(np.shape(binary_mask), 'float32')

    cube_list = cube_slicer(binary_mask, (32, 32, 32), (28, 28, 28))

    non_zero_cube_list = []

    for item in cube_list:
        if np.sum(item[1]) > 0:
            non_zero_cube_list.append(item)

    center_line_cube_list = Functions.func_parallel(basic_center_line_abstract, non_zero_cube_list)

    for item in center_line_cube_list:
        location = item[0]
        cube_center_line_array = item[1]

        center_line_array[location[0] + 2: location[0] + 30, location[1] + 2: location[1] + 30,
                          location[2] + 2: location[2] + 30] = cube_center_line_array[2: 30, 2: 30, 2: 30]

    return center_line_array


def refine_center_line_v2(raw_center_line, binary_mask):
    """
    refine follows these guidelines:
    1) extend raw_center_line by applying get outer surface by several times
    2) more_tube_like = extended_center_line * binary_mask
    3) apply "abstract_center_line" on more_tube_like
    4) repeat 1, 2, 3
    :param raw_center_line: the binary array in 3d numpy float32
    :param binary_mask: the binary_mask for semantic
    :return: the refined return_center_line array in numpy float32 same shape with the input
    """
    refined_center_line = np.array(raw_center_line, 'float32')

    bounding_box = Functions.get_bounding_box(raw_center_line, pad=3)

    center_line_tight = refined_center_line[bounding_box[0][0]: bounding_box[0][1],
                                            bounding_box[1][0]: bounding_box[1][1],
                                            bounding_box[2][0]: bounding_box[2][1]]

    for i in range(3):  # extending center line
        center_line_tight[:] = center_line_tight[:] + get_surface.get_surface(center_line_tight[:], True, False)

    refined_center_line = refined_center_line * binary_mask

    Functions.show_point_cloud_3d(refined_center_line, data_type='array')

    refined_center_line = abstract_center_line_3d(refined_center_line)

    Functions.show_point_cloud_3d(refined_center_line, data_type='array')

    return refined_center_line


def refine_circles_in_center_line(raw_center_line, distance_map, search_radius=4, max_parallel=24):
    """
    Remove voxels from small distance to large, if remove will not cause connectivity change in the search radius,
    remove it.
    :param max_parallel:
    :param search_radius: range to search connectivity change: the refined center line will not contain circle
    with diameter <= 2 * search_radius + 1.
    :param raw_center_line:
    :param distance_map:
    :return: refined_center_line
    """
    assert type(search_radius) is int
    assert 0 < search_radius < 12
    global search_range
    search_range = search_radius

    refined_center_line = np.zeros(np.shape(raw_center_line), 'float16')

    if search_radius > 0:
        shapes = np.shape(raw_center_line)
        raw_center_line[0: search_radius, :, :] = 0
        raw_center_line[:, 0: search_radius, :] = 0
        raw_center_line[:, :, 0: search_radius] = 0
        raw_center_line[(shapes[0] - search_radius):, :, :] = 0
        raw_center_line[:, (shapes[1] - search_radius):, :] = 0
        raw_center_line[:, :, (shapes[2] - search_radius):] = 0

    cube_list_raw = cube_slicer(raw_center_line, (64, 64, 64), (48, 48, 48))
    cube_list_distance = cube_slicer(distance_map, (64, 64, 64), (48, 48, 48))

    non_zero_cube_list = []

    for i in range(len(cube_list_raw)):
        item_raw = cube_list_raw[i]
        item_distance = cube_list_distance[i]
        if np.sum(item_raw[1]) > 0:
            non_zero_cube_list.append((item_raw[0], item_raw[1], item_distance[1]))

    center_line_cube_list = Functions.func_parallel(
        basic_refine_surface_remove_circle, non_zero_cube_list, parallel_count=max_parallel)

    for item in center_line_cube_list:
        location = item[0]
        cube_center_line_array = item[1]

        refined_center_line[location[0] + 8: location[0] + 56, location[1] + 8: location[1] + 56,
                            location[2] + 8: location[2] + 56] = cube_center_line_array[8: 56, 8: 56, 8: 56]

    return refined_center_line


def basic_center_line_abstract(input_tuple):
    """
    abstract the center line for the small cube
    :param input_tuple: with length == 2,
    first element is the location for the cube in the array (x_min, y_min, z_min), second is the tube in 3d numpy array
    :return:
    """

    return input_tuple[0], np.array(morphology.skeletonize_3d(input_tuple[1]) > 0, 'float32')


def basic_center_line_confirm(input_tuple):
    """
    check whether the center line voxel already good enough.
    criteria: part of the single line and of the extreme in the distance_map
    :param input_tuple: with length == 3,
    first element is the cube from return_center_line array centered with the voxel to be check, in shape 5x5x5
    second element is the cube from distance_map, in shape 5x5x5
    :return: True if the center voxel pass the criteria, else False
    """
    global loc_list_layer_one_voxel, loc_list_layer_two_voxel
    cube_center_line = input_tuple[0]  # voxel to be check in cube_center_line[2, 2, 2]
    cube_distance_map = input_tuple[1]

    # check whether on a single thread
    count_layer_one_center_line = 0
    for loc in loc_list_layer_one_voxel:
        count_layer_one_center_line += cube_center_line[loc]
    if count_layer_one_center_line > 2:
        return False
    count_layer_two_center_line = 0
    for loc in loc_list_layer_two_voxel:
        count_layer_two_center_line += cube_center_line[loc]
    if count_layer_two_center_line > 2:
        return False

    # check whether is the extreme point on distance map
    distance_to_surface_central = cube_distance_map[2, 2, 2]
    for loc in loc_list_layer_one_voxel:
        distance_to_surface = cube_distance_map[loc]
        if distance_to_surface > distance_to_surface_central and cube_center_line[loc] < 0.5:
            return False
    for loc in loc_list_layer_two_voxel:
        distance_to_surface = cube_distance_map[loc]
        if distance_to_surface > distance_to_surface_central and cube_center_line[loc] < 0.5:
            return False

    return True


def basic_remove_surface_voxel_temp(input_small_cube):
    """
    if remove the surface voxel will not increase the number of connected component, remove it
    :param input_small_cube: a 3 by 3 cube
    :return: True, remove the central voxel, False, leave the central voxel
    """

    remove_this_voxel = True

    if input_small_cube[0][0][0] == 1:
        lonely = True
        for loc in adjacent_list_000:
            if input_small_cube[loc] == 1:
                lonely = False  # voxel (0, 0, 0) is not only connected to the center
                break
        if lonely:
            remove_this_voxel = False  # only use the central voxel can retain the connectivity

    if input_small_cube[0][0][1] == 1:
        lonely = True
        for loc in adjacent_list_001:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][0][2] == 1:
        lonely = True
        for loc in adjacent_list_002:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][1][0] == 1:
        lonely = True
        for loc in adjacent_list_010:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][1][1] == 1:
        lonely = True
        for loc in adjacent_list_011:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][1][2] == 1:
        lonely = True
        for loc in adjacent_list_012:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][2][0] == 1:
        lonely = True
        for loc in adjacent_list_020:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][2][1] == 1:
        lonely = True
        for loc in adjacent_list_021:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[0][2][2] == 1:
        lonely = True
        for loc in adjacent_list_022:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][0][0] == 1:
        lonely = True
        for loc in adjacent_list_100:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][0][1] == 1:
        lonely = True
        for loc in adjacent_list_101:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][0][2] == 1:
        lonely = True
        for loc in adjacent_list_102:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][1][0] == 1:
        lonely = True
        for loc in adjacent_list_110:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][1][2] == 1:
        lonely = True
        for loc in adjacent_list_112:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][2][0] == 1:
        lonely = True
        for loc in adjacent_list_120:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][2][1] == 1:
        lonely = True
        for loc in adjacent_list_121:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[1][2][2] == 1:
        lonely = True
        for loc in adjacent_list_122:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][0][0] == 1:
        lonely = True
        for loc in adjacent_list_200:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][0][1] == 1:
        lonely = True
        for loc in adjacent_list_201:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][0][2] == 1:
        lonely = True
        for loc in adjacent_list_202:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][1][0] == 1:
        lonely = True
        for loc in adjacent_list_210:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][1][1] == 1:
        lonely = True
        for loc in adjacent_list_211:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][1][2] == 1:
        lonely = True
        for loc in adjacent_list_212:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][2][0] == 1:
        lonely = True
        for loc in adjacent_list_220:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][2][1] == 1:
        lonely = True
        for loc in adjacent_list_221:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    if input_small_cube[2][2][2] == 1:
        lonely = True
        for loc in adjacent_list_222:
            if input_small_cube[loc] == 1:
                lonely = False
                break
        if lonely:
            remove_this_voxel = False

    return remove_this_voxel


def basic_remove_voxel(input_small_cube):
    global search_range
    shape_cube = search_range * 2 + 1

    if not np.shape(input_small_cube) == (shape_cube, shape_cube, shape_cube):
        return False

    larger_array = np.zeros([shape_cube + 2, shape_cube + 2, shape_cube + 2], 'float32')
    larger_array[1:shape_cube + 1, 1:shape_cube + 1, 1:shape_cube + 1] = input_small_cube[:]
    original_components = len(connectivity.get_connected_regions(larger_array, strict=False)[1])
    larger_array[search_range + 1, search_range + 1, search_range + 1] = 0
    current_components = len(connectivity.get_connected_regions(larger_array, strict=False)[1])

    if current_components > original_components:
        return False
    return True


def basic_refine_surface_remove_circle(input_tuple):
    global search_range

    cube_binary_mask = input_tuple[1]
    cube_distance_map = input_tuple[2]

    max_distance, min_distance = np.max(cube_distance_map), np.min(cube_distance_map)

    for current_distance in range(int(min_distance) + 1, int(max_distance) + 1):

        loc_list_current_distance = Functions.get_location_list(np.where(cube_distance_map == current_distance))

        for loc in loc_list_current_distance:
            if loc[0] < (search_range + 1) or (loc[1] < search_range + 1) or (loc[2] < search_range + 1) or \
                    loc[0] > (63 - search_range) or loc[1] > (63 - search_range) or loc[2] > (63 - search_range):
                continue
            if basic_remove_voxel(
                    cube_binary_mask[loc[0] - search_range: loc[0] + search_range + 1,
                                     loc[1] - search_range: loc[1] + search_range + 1,
                                     loc[2] - search_range: loc[2] + search_range + 1]):
                cube_binary_mask[loc] = 0

    return input_tuple[0], cube_binary_mask


def cube_slicer(original_array, cube_size, step):
    """
    slice the original_array into list of cubes
    :param original_array: array to be slice
    :param cube_size: a tuple, the shape for cubes, len(cube_size) == len(shape(original_array))
    :param step: a tuple, the step on each axis
    :return: list of cubes with initial loc, [((x_1, y_1, z_1, ...), cube_1), ...]
    Note:           cube_1 = original_array[x_1: x_1 + cube_size[0], ...]
    """
    shape_original = np.shape(original_array)
    assert len(shape_original) == len(cube_size)
    assert len(shape_original) == len(step)
    assert 0 < len(shape_original) < 5

    shape_padded = list(shape_original)
    for axis in range(len(shape_padded)):
        if shape_padded[axis] % int(cube_size[axis]) > 0:
            shape_padded[axis] += int(cube_size[axis]) - shape_padded[axis] % int(cube_size[axis])
        shape_padded[axis] += step[axis]

    padded_original_array = np.zeros(shape_padded, 'float32')

    if len(shape_padded) == 1:
        padded_original_array[0: shape_original[0]] = original_array
    elif len(shape_padded) == 2:
        padded_original_array[0: shape_original[0], 0: shape_original[1]] = original_array
    elif len(shape_padded) == 3:
        padded_original_array[0: shape_original[0], 0: shape_original[1], 0: shape_original[2]] = original_array
    elif len(shape_padded) == 4:
        padded_original_array[0: shape_original[0], 0: shape_original[1], 0: shape_original[2], 0: shape_original[3]] \
            = original_array

    list_start_loc = []

    if len(shape_padded) == 1:
        for i in range(int(shape_original[0] / step[0]) + 1):
            if i * step[0] >= shape_original[0]:
                break
            list_start_loc.append((int(i * step[0])))

    if len(shape_padded) == 2:
        for i in range(int(shape_original[0] / step[0]) + 1):
            if i * step[0] >= shape_original[0]:
                break
            for j in range(int(shape_original[1] / step[1]) + 1):
                if j * step[1] >= shape_original[1]:
                    break
                list_start_loc.append((int(i * step[0]), int(j * step[1])))

    if len(shape_padded) == 3:
        for i in range(int(shape_original[0] / step[0]) + 1):
            if i * step[0] >= shape_original[0]:
                break
            for j in range(int(shape_original[1] / step[1]) + 1):
                if j * step[1] >= shape_original[1]:
                    break
                for k in range(int(shape_original[2] / step[2]) + 1):
                    if k * step[2] >= shape_original[2]:
                        break
                    list_start_loc.append((int(i * step[0]), int(j * step[1]), int(k * step[2])))

    if len(shape_padded) == 4:
        for i in range(int(shape_original[0] / step[0]) + 1):
            if i * step[0] >= shape_original[0]:
                break
            for j in range(int(shape_original[1] / step[1]) + 1):
                if j * step[1] >= shape_original[1]:
                    break
                for k in range(int(shape_original[2] / step[2]) + 1):
                    if k * step[2] >= shape_original[2]:
                        break
                    for l in range(int(shape_original[3] / step[3]) + 1):
                        if l * step[3] >= shape_original[3]:
                            break
                        list_start_loc.append((int(i * step[0]), int(j * step[1]), int(k * step[2]), int(l * step[3])))

    loc_tube_list = []
    if len(shape_padded) == 1:
        for loc in list_start_loc:
            loc_tube_list.append((loc, original_array[loc[0]: loc[0] + cube_size[0]]))
    if len(shape_padded) == 2:
        for loc in list_start_loc:
            loc_tube_list.append((loc, original_array[loc[0]: loc[0] + cube_size[0], loc[1]: loc[1] + cube_size[1]]))
    if len(shape_padded) == 3:
        for loc in list_start_loc:
            loc_tube_list.append((loc, original_array[loc[0]: loc[0] + cube_size[0], loc[1]: loc[1] + cube_size[1],
                                 loc[2]: loc[2] + cube_size[2]]))
    if len(shape_padded) == 4:
        for loc in list_start_loc:
            loc_tube_list.append((loc, original_array[loc[0]: loc[0] + cube_size[0], loc[1]: loc[1] + cube_size[1],
                                 loc[2]: loc[2] + cube_size[2], loc[3]: loc[3] + cube_size[3]]))
    return loc_tube_list


def get_center_line(binary_mask, threshold=None, surface_distance=None, search_radius=4, return_dtype='float32',
                    max_parallel_count=48):
    """

    :param max_parallel_count:
    :param return_dtype: data type of return array
    :param binary_mask: mask to extract center line
    :param threshold:
    :param surface_distance: the return of function "get_surface_distance(binary_mask)", None if search_radius is None.
    :param search_radius: if the center line should not contain circles, apply this to remove circles. the refined
    center line will not contain circles with radius < 2 * search_radius + 1.
    :return: binary array, same shape with the input, in numpy return_dtype
    """

    if threshold is not None:
        binary_mask = np.array(binary_mask > threshold, 'uint8')
    if not binary_mask.dtype == 'uint8':
        binary_mask = np.array(binary_mask, 'uint8')

    if search_radius is None:
        assert surface_distance is None
        return abstract_center_line_3d(binary_mask)

    else:
        if surface_distance is None:
            surface_distance = get_surface_distance(binary_mask)
        raw_center_line = abstract_center_line_3d(binary_mask)

        center_line_refined = refine_circles_in_center_line(
            raw_center_line, surface_distance, search_radius=search_radius, max_parallel=max_parallel_count)

        return np.array(center_line_refined, return_dtype)


def pipeline_get_depth_array_and_center_line_array(top_dict_blood_vessel_mask, top_dict_save, fold=(0, 1)):
    """

    :param top_dict_blood_vessel_mask:
    :param top_dict_save:
    :param fold:
    :return: None
    """
    import os
    fn_list = os.listdir(top_dict_blood_vessel_mask)[fold[0]::fold[1]]

    processed = 0

    for fn in fn_list:

        blood_vessel_mask = np.load(os.path.join(top_dict_blood_vessel_mask, fn))['array']

        depth_array = get_surface_distance(blood_vessel_mask)

        center_line_array = get_center_line(depth_array, surface_distance=depth_array)

        Functions.save_np_array(
            os.path.join(top_dict_save, 'depth_and_center-line/center_line_mask/'), fn[:-4], center_line_array, True)
        Functions.save_np_array(
            os.path.join(top_dict_save, 'depth_and_center-line/depth_array/'), fn[:-4], depth_array, True)


if __name__ == '__main__':

    pipeline_get_depth_array_and_center_line_array('/media/zhoul0a/New Volume/RAD-ChestCT_dataset/semantic_in_rescaled_ct/blood_mask/',
                                                   '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/', (0, 3))
    exit()
    blood_mask = Functions.read_in_mha(
        '/home/zhoul0a/Desktop/absolutely_normal/rescaled_CT_central_axis/Scanner-A_A1_predict_artery.mha')
    surface_distance_array = get_surface_distance(blood_mask)
    for i in range(100, 400, 10):
        Functions.image_show(surface_distance_array[:, :, i])

    exit()
    rescaled_array = np.load('/home/zhoul0a/Desktop/absolutely_normal/rescaled_ct/Scanner-A_A1.npy')

    blood_mask = np.array(blood_mask, 'uint8')
    basic_center_line = abstract_center_line_3d(blood_mask)
    Functions.show_point_cloud_3d(basic_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), basic_center_line[:, :, 256])

    surface_distance_array = get_surface_distance(blood_mask)

    new_center_line = refine_circles_in_center_line(basic_center_line, surface_distance_array, search_radius=3)
    Functions.show_point_cloud_3d(new_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), new_center_line[:, :, 256])
    new_center_line = refine_circles_in_center_line(basic_center_line, surface_distance_array, search_radius=4)
    Functions.show_point_cloud_3d(new_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), new_center_line[:, :, 256])
    new_center_line = refine_circles_in_center_line(basic_center_line, surface_distance_array, search_radius=5)
    Functions.show_point_cloud_3d(new_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), new_center_line[:, :, 256])
    new_center_line = refine_circles_in_center_line(basic_center_line, surface_distance_array, search_radius=6)
    Functions.show_point_cloud_3d(new_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), new_center_line[:, :, 256])
    new_center_line = refine_circles_in_center_line(basic_center_line, surface_distance_array, search_radius=7)
    Functions.show_point_cloud_3d(new_center_line)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), new_center_line[:, :, 256])
