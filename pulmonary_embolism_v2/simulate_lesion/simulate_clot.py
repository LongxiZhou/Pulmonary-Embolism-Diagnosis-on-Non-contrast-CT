"""

clot_sample_dict: a dict, with key "loc_depth_set" and "range_clot"
clot_sample_dict["loc_clot_set"] = {(x, y, z), }
clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, ..., 'max_depth': max_depth}  here b is the clot depth
the mass center for the location x, y, z is (0, 0, 0)
clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations


sample_sequence_copy: a list, each item is a dict:
{'ct_data': ct_cube in np_array, 'penalty_weight': None, 'location_offset': central_location_offset, e.g., (23, 12, 9),
'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'clot_depth': clot_depth,
'clot_array': None}

"""

import numpy as np
import copy
import random
from functools import partial
import Tool_Functions.Functions as Functions


def apply_clot_on_sample_sequence(sample_sequence, clot_sample_dict, func_change_ct, power_range=(-0.3, 0.6),
                                  add_base_range=(0, 3), value_increase=(0.01, 0.02), voxel_variance=(0.5, 1)):
    """

    :param value_increase:
    :param power_range:
    :param add_base_range:
    :param sample_sequence:
    :param clot_sample_dict:
    :param func_change_ct: func_increase_ct(clot_depth, add_base, power),
    return a float, will be added to the ct_cube at the location
    :param voxel_variance:
    :return: sample_sequence added with blood clot
    """

    sample_sequence_copy = copy.deepcopy(sample_sequence)  # copy this sequence

    index_clot_center = random.randint(0, len(sample_sequence_copy) - 1)
    center_location_clot = sample_sequence_copy[index_clot_center]["center_location"]
    # sample_sequence_copy[index_clot_center] is selected as the cube lays in the mass center of clot

    new_clot_loc_set = get_new_loc_set(center_location_clot, clot_sample_dict)
    # update the location_set of where has clot

    potential_sample_list = select_potential_cubes(sample_sequence_copy, index_clot_center, clot_sample_dict)
    # use bounding box to reduce searching range

    depth_power = random.uniform(power_range[0], power_range[1])
    depth_add_base = random.uniform(add_base_range[0], add_base_range[1])

    increase_factor = random.uniform(value_increase[0], value_increase[1])

    func_increase_ct = partial(func_change_ct, add_base=depth_add_base, power=depth_power)

    clot_depth_dict = clot_sample_dict["clot_depth_dict"]
    max_clot_depth = clot_depth_dict['max_depth'] + depth_add_base

    # print("depth_power:", depth_power, "depth_add_base:", depth_add_base, 'max_clot_depth:', max_clot_depth)

    max_possible_value = max(func_increase_ct(1), func_increase_ct(max_clot_depth))  # use this to normalize

    num_clot_voxels = change_ct_value_and_establish_clot_array(potential_sample_list, new_clot_loc_set, clot_depth_dict,
                                                               center_location_clot, func_increase_ct,
                                                               max_possible_value, increase_factor, voxel_variance, 
                                                               max_clot_depth)

    return sample_sequence_copy, num_clot_voxels


def get_new_loc_set(center_location_clot, clot_sample_dict):
    initial_loc_set = clot_sample_dict["loc_clot_set"]

    x_c, y_c, z_c = center_location_clot

    new_loc_set = set()

    for location in initial_loc_set:
        new_loc_set.add((location[0] + x_c, location[1] + y_c, location[2] + z_c))

    return new_loc_set


def select_potential_cubes(sample_sequence_copy, index_clot_center, clot_sample_dict):
    """

    :param sample_sequence_copy:
    :param index_clot_center:
    :param clot_sample_dict:
    :return:
    """
    center_location_clot = sample_sequence_copy[index_clot_center]["center_location"]

    cub_shape = np.shape(sample_sequence_copy[0]["ct_data"])  # like (5, 5, 5)
    x_radius = int(cub_shape[0] / 2)
    y_radius = int(cub_shape[1] / 2)
    z_radius = int(cub_shape[2] / 2)

    range_clot = clot_sample_dict["range_clot"]

    bounding_box_x = (center_location_clot[0] + range_clot[0][0], center_location_clot[0] + range_clot[0][1])
    bounding_box_y = (center_location_clot[1] + range_clot[1][0], center_location_clot[1] + range_clot[1][1])
    bounding_box_z = (center_location_clot[2] + range_clot[2][0], center_location_clot[2] + range_clot[2][1])

    potential_sample_list = []

    for sample in sample_sequence_copy:
        x_center, y_center, z_center = sample["center_location"]
        if bounding_box_x[0] < (x_center + x_radius) and (x_center - x_radius) < bounding_box_x[1]:
            if bounding_box_y[0] < (y_center + y_radius) and (y_center - y_radius) < bounding_box_y[1]:
                if bounding_box_z[0] < (z_center + z_radius) and (z_center - z_radius) < bounding_box_z[1]:
                    potential_sample_list.append(sample)

    return potential_sample_list


def change_ct_value_and_establish_clot_array(potential_sample_list, loc_clot_set, clot_depth_dict, center_location_clot,
                                             func_increase_ct, max_possible_increase, increase_factor, voxel_variance,
                                             max_clot_depth):
    """

    :param max_clot_depth: max_clot_depth. small depth means lower increase value
    :param potential_sample_list: a sample from the sequence
    :param loc_clot_set: return from function "get_new_loc_set"
    :param clot_depth_dict
    :param center_location_clot
    :param func_increase_ct:
    :param max_possible_increase: the max_possible_increase for func_increase_ct
    :param increase_factor
    :param voxel_variance
    :return: num_voxel_changed  # the number of clot voxels
    """
    assert len(potential_sample_list) > 0
    cub_shape = np.shape(potential_sample_list[0]['ct_data'])
    assert cub_shape[0] % 2 == 1 and cub_shape[1] % 2 and cub_shape[2] % 2

    x_radius = int(cub_shape[0] / 2)
    y_radius = int(cub_shape[1] / 2)
    z_radius = int(cub_shape[2] / 2)

    clot_center_x, clot_center_y, clot_center_z = center_location_clot

    num_voxel_changed = np.array([0, ], 'int32')  # the number of clot voxels

    def process_one_sample(sample_input):
        ct_cube = sample_input['ct_data']
        blood_vessel_depth_array = sample_input['depth_cube']
        loc_set_sample = set()
        x_c, y_c, z_c = sample_input['center_location']
        for x in range(-x_radius, x_radius + 1):
            for y in range(-y_radius, y_radius + 1):
                for z in range(-z_radius, z_radius + 1):
                    loc_set_sample.add((x_c + x, y_c + y, z_c + z))

        intersection_loc_set = loc_set_sample & loc_clot_set  # the absolute locations

        if len(intersection_loc_set) > 0:
            clot_array = np.zeros(cub_shape, 'float32')

            for location in intersection_loc_set:
                local_loc = (location[0] - x_c + x_radius, location[1] - y_c + y_radius, location[2] - z_c + z_radius)
                if blood_vessel_depth_array[local_loc] < 4:
                    continue

                relative_loc = (location[0] - clot_center_x, location[1] - clot_center_y, location[2] - clot_center_z)
                clot_depth = clot_depth_dict[relative_loc]

                add_value = func_increase_ct(clot_depth) / max_possible_increase * increase_factor * random.uniform(
                    voxel_variance[0], voxel_variance[1]) * random.uniform(clot_depth / max_clot_depth, 1)

                ct_cube[local_loc] = ct_cube[local_loc] + add_value
                clot_array[local_loc] = add_value

                num_voxel_changed[0] = num_voxel_changed[0] + 1

            sample_input['clot_array'] = clot_array

    for sample in potential_sample_list:
        process_one_sample(sample)

    return num_voxel_changed[0]


def func_change_ct_1(clot_depth, add_base, power):
    return (clot_depth + add_base) ** power


def func_change_ct_test(clot_depth, add_base, power):
    return 1


def visualize_simulated_clot(clot_sample_dict):
    bounding_box = clot_sample_dict["range_clot"]
    loc_set = clot_sample_dict["loc_clot_set"]
    print("length loc set:", len(loc_set))
    depth_dict = clot_sample_dict["clot_depth_dict"]

    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box

    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')

    for loc in loc_set:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = depth_dict[loc]

    new_array = np.array(temp_array > 0.5, 'float32')
    import analysis.center_line_and_depth_3D as get_depth
    import visualization.visualize_3d.visualize_stl as stl

    """
    new_array = connect_region_detect.convert_to_simply_connected_old(new_array, dimension=3, add_outer_layer=0,
                                                                      return_array_dtype='float32')
    """

    stl.visualize_numpy_as_stl(new_array)

    new_depth_array = get_depth.get_surface_distance(new_array)

    print("max encoding_depth", np.max(new_depth_array))

    Functions.image_show(new_depth_array[:, :, int((z_max - z_min + 4) / 2)])


def check_correctness_2(clot_sample_dict):
    bounding_box = clot_sample_dict["range_clot"]
    loc_set = clot_sample_dict["loc_clot_set"]
    print("length loc set:", len(loc_set))
    depth_dict = clot_sample_dict["clot_depth_dict"]

    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box

    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')

    for loc in loc_set:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = depth_dict[loc]

    import visualization.visualize_3d.visualize_stl as stl

    stl.visualize_numpy_as_stl(np.array(temp_array > 0.5, 'float32'))

    print("max encoding_depth", np.max(temp_array))

    Functions.image_show(temp_array[:, :, int((z_max - z_min + 4) / 2)])


def check_correctness_3(clot_sample_dict, sample_sequence):
    import pulmonary_embolism.sequence_rescaled_ct_converter as converter
    import visualization.visualize_3d.visualize_stl as stl

    new_sample_sequence, clot_count = \
        apply_clot_on_sample_sequence(sample_sequence, clot_sample_dict, func_change_ct_test)
    print("there are:", clot_count, 'voxels for clot')

    clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(new_sample_sequence, (4, 4, 5), key='clot_array')

    ct_array = converter.reconstruct_rescaled_ct_from_sample_sequence(new_sample_sequence, (4, 4, 5), key='ct_data')

    ct_array_raw = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, (4, 4, 5), key='ct_data')

    stl.visualize_numpy_as_stl(clot_mask)

    bounding_box = Functions.get_bounding_box(clot_mask)

    for z in range(bounding_box[2][0], bounding_box[2][1], 5):
        image_left = Functions.cast_to_0_1(ct_array_raw[:, :, z])
        image_right = Functions.cast_to_0_1(ct_array[:, :, z])
        image = np.concatenate((image_left, image_right), axis=1)
        Functions.image_show(image)


def check_correctness_4(clot_sample_dict, sample_sequence):
    import pulmonary_embolism.sequence_rescaled_ct_converter as converter
    import visualization.visualize_3d.visualize_stl as stl

    new_sample_sequence, clot_count = \
        apply_clot_on_sample_sequence(sample_sequence, clot_sample_dict, func_change_ct_1, power_range=(-0, 0.01),
                                      add_base_range=(0, 3), value_increase=(0.023, 0.046), voxel_variance=(0.99, 1))
    print("there are:", clot_count, 'voxels for clot')

    clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(new_sample_sequence, (4, 4, 5), key='clot_array')

    clot_mask = np.array(clot_mask > 0, 'float32')

    ct_array = converter.reconstruct_rescaled_ct_from_sample_sequence(new_sample_sequence, (4, 4, 5), key='ct_data')

    ct_array_raw = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, (4, 4, 5), key='ct_data')

    stl.visualize_numpy_as_stl(clot_mask)

    bounding_box = Functions.get_bounding_box(clot_mask)

    for z in range(bounding_box[2][0], bounding_box[2][1], 5):
        image_left = Functions.cast_to_0_1(ct_array[:, :, z] - ct_array_raw[:, :, z])
        image_mid = Functions.cast_to_0_1(ct_array[:, :, z])
        image_right = Functions.cast_to_0_1(np.clip(Functions.change_to_HU(ct_array[:, :, z]), -100, 300))
        image = np.concatenate((image_left, image_mid, image_right), axis=1)
        Functions.image_show(image)


if __name__ == '__main__':
    from med_transformer.image_transformer.transformer_for_PE.generate_clot_samlpe_dict import get_clot_sample_dict_from_loc_list

    temp_clot_sample_list = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/list-clot_sample_dict/Scanner-B-B22_reduced.pickle')
    print(len(temp_clot_sample_list))

    temp_clot_sample = temp_clot_sample_list[0]

    sample_sequence_test = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/reduced_merged-refine_length-3000_branch-7/disk10-6_2020-04-28.pickle'
    )

    check_correctness_4(temp_clot_sample, sample_sequence_test)
    check_correctness_4(temp_clot_sample, sample_sequence_test)
    check_correctness_4(temp_clot_sample, sample_sequence_test)

    exit()

    print(temp_clot_sample["range_clot"])

    test_array = np.zeros([50, 50, 50], 'float32')

    clot_loc_set = temp_clot_sample["loc_clot_set"]
    clot_depth_dict = temp_clot_sample['depth_dict']

    for loc in clot_loc_set:
        test_array[int(25 + loc[0]), int(25 + loc[1]), int(25 + loc[2])] = clot_depth_dict[loc]

    for z in range(20, 30):
        Functions.image_show(test_array[:, :, z])

    exit()



    exit()

    temp_clot = np.zeros([80, 80, 80], 'float32')

    temp_clot[10: 70, 10: 70, 10: 70] = 1

    clot_loc_list = Functions.get_location_list(np.where(temp_clot > 0.5))

    temp_clot_sample = get_clot_sample_dict_from_loc_list(clot_loc_list)

    sample_sequence_test = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/merged_reduced/disk10-6_2020-04-28.pickle'
    )

    check_correctness_3(temp_clot_sample, sample_sequence_test)

    exit()

    list_sample_dict_reduced = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/temp/Scanner-A-A12.pickle')

    for i in range(0, len(list_sample_dict_reduced)):
        print(i)
        print(list_sample_dict_reduced[i]['range_clot'])



    exit()
