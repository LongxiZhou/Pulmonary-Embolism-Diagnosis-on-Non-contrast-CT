import numpy as np
import analysis.get_surface_rim_adjacent_mean as get_surface_and_rim
import analysis.connect_region_detect as connect_region_detect
import Tool_Functions.Functions as Functions
import random


def convert_to_simply_connected(stack_region_mask, dimension=2, parallel_count=None, iter_round=2, add_outer_layer=1):
    """

    :param add_outer_layer:
    :param iter_round:
    :param stack_region_mask: numpy array in shape [batch, x, y] or [x, y] or [batch, x, y, z] or [x, y, z]

        for each region_mask, there should be only one connect region

    :param dimension: the dimension of the region mask, 2 or 3
    :param parallel_count: None for not parallel, int for max parallel count. The parallel is on batch level
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
            array[:, :, shape_array[1] - 1] = 0

    if len(shape_stack) > dimension:
        for slice_id in range(shape_stack[0]):
            trim_boundary_to_zero(stack_region_mask[slice_id])
    else:
        trim_boundary_to_zero(stack_region_mask)

    if dimension == 2:
        rim_or_surface = get_surface_and_rim.get_rim(stack_region_mask, outer=True, strict=False)
    else:
        rim_or_surface = get_surface_and_rim.get_surface(stack_region_mask, outer=True, strict=False)

    return_array = np.array(stack_region_mask, 'float16')

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

    return convert_to_simply_connected(return_array, dimension, parallel_count, iter_round - 1, 0)


def derive_topological_connectivity_mask(input_tuple):

    rim_or_surface_mask, original_image = input_tuple
    type_sorted_loc_dict = connect_region_detect.get_connected_regions_discrete(rim_or_surface_mask, strict=True)
    sorted_loc_dict = type_sorted_loc_dict[1]
    num_regions = len(sorted_loc_dict)
    for inside_region_id in range(2, num_regions + 1):
        location_list = sorted_loc_dict[inside_region_id]
        for location in location_list:
            original_image[location] = 1
    return original_image


def get_depth_2d(binary_image, strict=True):
    depth_array = np.zeros(np.shape(binary_image), 'int32')
    temp_image = np.array(binary_image, 'float32')
    depth = 1
    while np.sum(temp_image) > 0:
        rim = get_surface_and_rim.get_rim(temp_image, outer=False, strict=strict)
        depth_array = depth_array + depth * rim
        depth += 1
        temp_image = temp_image - rim
    return depth_array


def set_mass_center(location_list_or_array, center_location=(256, 256), return_loc_list=True, bounding_box=None):

    if bounding_box is not None:  # the index range of the axis, like ((20, 500), (20, 500), (20, 500))
        assert len(bounding_box) == len(center_location)

    if type(location_list_or_array[0]) is list or type(location_list_or_array[0]) is tuple:
        location_array = Functions.get_location_array(location_list_or_array)
    else:
        location_array = location_list_or_array
    current_mass_center = []
    for projection_array in location_array:
        current_mass_center.append(int(np.average(projection_array)))

    location_array_new = []
    for i, projection_array in enumerate(location_array):
        projection_array = projection_array + int(center_location[i] - current_mass_center[i])
        if bounding_box is not None:
            projection_array = np.clip(projection_array, bounding_box[i][0], bounding_box[i][1])
        location_array_new.append(projection_array)
    if return_loc_list:
        return Functions.get_location_list(location_array_new)
    return tuple(location_array_new)


def random_rotate_scale_and_move(location_list_or_array, shape_return_array=(512, 512), rotate_range=(0, 360),
                                 scale_range=(0.25, 4), pad_distance=(20, 20), return_binary=False, show=False):
    """

    :param location_list_or_array: the source of the pattern
    :param shape_return_array:
    :param rotate_range: degree of rotate, sample from uniform(rotate_range[0], rotate_range[1])
    :param scale_range:
    determine the scale:
        sample x from uniform(-1/scale_range[0] + 1, scale_range[1] - 1), then map to scale_range.
    :param pad_distance:
    determine the mass center:
        mass_center[i] is from uniform(pad_distance[i], shape_return_array[i] - pad_distance[i])
    :param  return_binary: whether the return array retain the value of scaled on boundary
    :param show
    :return: pattern in numpy_float32, min 0, max 1.
    """
    assert 0 < scale_range[0] < scale_range[1]
    if scale_range[0] < 1 and scale_range[1] < 1:
        scale_sample = random.uniform(-1/scale_range[0] + 1, -1/scale_range[1] + 1)
    elif scale_range[0] < 1 and scale_range[1] >= 1:
        scale_sample = random.uniform(-1/scale_range[0] + 1, scale_range[1] - 1)
    else:
        scale_sample = random.uniform(scale_range[0] - 1, scale_range[1] - 1)

    temp_array = np.zeros(shape_return_array, 'float32')

    if scale_sample >= 0:
        scale_factor = scale_sample + 1
    else:
        scale_factor = 1 / (1 - scale_sample)

    rotate_degree = random.uniform(rotate_range[0], rotate_range[1])
    final_mass_center = (random.uniform(pad_distance[0], shape_return_array[0] - pad_distance[0]),
                         random.uniform(pad_distance[1], shape_return_array[1] - pad_distance[1]))

    if show:
        print("scale factor:", scale_factor, "rotate degree:", rotate_degree, "final_mass_center", final_mass_center)

    bounding_box = ((0, shape_return_array[0] - 1), (0, shape_return_array[1] - 1))

    initial_center = (int(shape_return_array[0] / 2), int(shape_return_array[1] / 2))

    loc_array_set_to_center = set_mass_center(location_list_or_array, initial_center, False, bounding_box)

    temp_array[loc_array_set_to_center] = 1

    temp_array = Functions.rotate_and_scale_image(temp_array, rotate_degree, initial_center, scale_factor)

    loc_array_greater_zero = np.where(temp_array > 0)

    if return_binary:
        loc_array_final = set_mass_center(loc_array_greater_zero, final_mass_center, False, bounding_box)
        return_array = np.zeros(shape_return_array, 'float32')
        return_array[loc_array_final] = 1
        return return_array

    loc_list_final = set_mass_center(loc_array_greater_zero, final_mass_center, True, bounding_box)
    return_array = np.zeros(shape_return_array, 'float32')

    for count, location in enumerate(loc_list_final):
        return_array[location] = temp_array[loc_array_greater_zero[0][count], loc_array_greater_zero[1][count]]

    return return_array


def convert_loc_depth_to_pattern(source_arrays, shape_return_array=(512, 512), rotate_range=(0, 360),
                                 scale_range=(0.25, 4), pad_distance=(20, 20), depth_power_range=(-1, 1),
                                 depth_add_base_range=(0, 3), show=False, random_seed=None):
    """

    :param random_seed:
    :param source_arrays: the source of the pattern
            (location_array, location_array, depth_array)
    :param shape_return_array:
    :param rotate_range: degree of rotate, sample from uniform(rotate_range[0], rotate_range[1])
    :param scale_range:
    determine the scale:
        sample x from uniform(-1/scale_range[0] + 1, scale_range[1] - 1), then map to scale_range.

    the value for pattern will be np.power(source_arrays[2] + add_base, power_for_depth), then set max to 1
    :param depth_power_range: get "power_for_depth"
    :param depth_add_base_range: get "add_base"

    :param pad_distance:
    determine the mass center:
        mass_center[i] is from uniform(pad_distance[i], shape_return_array[i] - pad_distance[i])
    :param show
    :return: pattern in numpy_float32, min 0, max 1.
    """
    if random_seed is not None:
        random.seed(random_seed)
    assert 0 < scale_range[0] < scale_range[1]
    if scale_range[0] < 1 and scale_range[1] < 1:
        scale_sample = random.uniform(-1/scale_range[0] + 1, -1/scale_range[1] + 1)
    elif scale_range[0] < 1 and scale_range[1] >= 1:
        scale_sample = random.uniform(-1/scale_range[0] + 1, scale_range[1] - 1)
    else:
        scale_sample = random.uniform(scale_range[0] - 1, scale_range[1] - 1)

    temp_array = np.zeros(shape_return_array, 'float32')

    if scale_sample >= 0:
        scale_factor = scale_sample + 1
    else:
        scale_factor = 1 / (1 - scale_sample)

    rotate_degree = random.uniform(rotate_range[0], rotate_range[1])
    power_for_depth = random.uniform(depth_power_range[0], depth_power_range[1])
    add_base = random.uniform(depth_add_base_range[0], depth_add_base_range[1])
    final_mass_center = (random.uniform(pad_distance[0], shape_return_array[0] - pad_distance[0]),
                         random.uniform(pad_distance[1], shape_return_array[1] - pad_distance[1]))

    if show:
        print("scale factor:", scale_factor, "rotate degree:", rotate_degree, "final_mass_center", final_mass_center,
              "power for encoding_depth:", power_for_depth, "add base:", add_base)

    bounding_box = ((0, shape_return_array[0] - 1), (0, shape_return_array[1] - 1))

    initial_center = (int(shape_return_array[0] / 2), int(shape_return_array[1] / 2))

    location_array = (source_arrays[0], source_arrays[1])

    loc_array_set_to_center = set_mass_center(location_array, initial_center, False, bounding_box)

    value_array = np.power(source_arrays[2] + add_base, power_for_depth)
    value_array = value_array / np.max(value_array)

    temp_array[loc_array_set_to_center] = value_array

    temp_array = Functions.rotate_and_scale_image(temp_array, rotate_degree, initial_center, scale_factor)

    loc_array_greater_zero = np.where(temp_array > 0)

    loc_list_final = set_mass_center(loc_array_greater_zero, final_mass_center, True, bounding_box)
    return_array = np.zeros(shape_return_array, 'float32')

    for count, location in enumerate(loc_list_final):
        return_array[location] = temp_array[loc_array_greater_zero[0][count], loc_array_greater_zero[1][count]]

    return return_array


if __name__ == '__main__':
    object_pickle = Functions.pickle_load_object('/home/zhoul0a/Desktop/denoise_project/source_pickle_with_depth/3.7K_high_resolution.pickle')
    for item in object_pickle:
        image = convert_loc_depth_to_pattern(item, show=True)
        Functions.image_show(image)


    exit()
    import os
    fn_list = os.listdir('/home/zhoul0a/Desktop/denoise_project/source_pickle/')

    for fn in fn_list:
        print(fn)
        object_pickle = Functions.pickle_load_object(
            '/home/zhoul0a/Desktop/denoise_project/source_pickle/' + fn)
        new_object = []

        for item in object_pickle:
            image = np.zeros([512, 512], 'int16')
            image[item] = 1

            depth_array = get_depth_2d(image, strict=False)

            test_array = np.zeros((len(item[0]),), 'int16')

            loc_list = Functions.get_location_list(item)
            for i, loc in enumerate(loc_list):
                test_array[i] = depth_array[loc]

            # image[item] = test_array

            new_item = (item[0], item[1], test_array)

            new_object.append(new_item)

        new_object = tuple(new_object)

        Functions.pickle_save_object('/home/zhoul0a/Desktop/denoise_project/source_pickle_with_depth/' + fn, new_object)

    exit()

    for item in object_pickle:
        image = random_rotate_scale_and_move(item)
        Functions.image_show(image)


    exit()

    test_array = np.load('/home/zhoul0a/Desktop/prognosis_project/Ising_model/ising_pattern_high_resolution/4.7_ising_high_resolution.npy')
    # test_array = np.load('/home/zhoul0a/Desktop/prognosis_project/Ising_model/ising_pattern_refine/4.0_ising_1.npy')
    sub_array = test_array#[0: 512, 0: 512]  #[1000:2000, 1000:2000]

    Functions.image_show(sub_array)

    id_sorted_loc_dict = connect_region_detect.get_connected_regions_discrete(sub_array, strict=True)

    pattern_source_list = []

    for id_key, sorted_dict in id_sorted_loc_dict.items():
        print("semantic:", id_key)
        """
        for region_id, loc_list in sorted_dict.items():
            print(region_id, len(loc_list))
        """

        for region_id, loc_list in sorted_dict.items():
            if 400 < len(loc_list) < 4000:
                """"""
                image = np.zeros([512, 512], 'float32')

                print("semantic", id_key, "showing:", region_id, "length:", len(loc_list))

                loc_array = set_mass_center(loc_list, (256, 256), return_loc_list=False)

                image[loc_array] = 1

                image = convert_to_simply_connected(image, add_outer_layer=1)

                loc_array_new = np.where(image > 0)

                loc_array_save = (np.array(loc_array_new[0], 'int16'), np.array(loc_array_new[1], 'int16'))

                pattern_source_list.append(loc_array_save)

                # image = random_rotate_scale_and_move(loc_array_save, show=True)


    pattern_source_tuple = tuple(pattern_source_list)
    print("final sample number:", len(pattern_source_tuple))
    Functions.pickle_save_object('/home/zhoul0a/Desktop/denoise_project/source_pickle/4.7K_high_resolution.pickle', pattern_source_tuple)

