"""

clot_sample_dict: a dict, with key "loc_depth_set" and "range_clot"
clot_sample_dict["loc_clot_set"] = {(x, y, z), }
clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, }  here b is the branching level
the mass center for the location x, y, z is (0, 0, 0)
clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations


sample_sequence_copy: a list, each item is a dict:
{'ct_data': ct_cube in np_array, 'penalty_weight': None, 'location_offset': central_location_offset, e.g., (23, 12, 9),
'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'clot_depth': clot_depth,
'clot_array': None, 'branch_array': None}

"""

import numpy as np
import copy
import random
from functools import partial


def apply_clot_on_sample_sequence(sample_sequence, clot_sample_dict, func_change_ct, branch_power_range=(-1, 1),
                                  branch_add_base_range=(0, 3)):
    """

    :param branch_power_range:
    :param branch_add_base_range:
    :param sample_sequence:
    :param clot_sample_dict:
    :param func_change_ct: func_increase_ct(clot_depth), return a float, will be added to the ct_cube at the location
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

    depth_dict = clot_sample_dict["clot_depth_dict"]

    branch_power = random.uniform(branch_power_range[0], branch_power_range[1])
    branch_add_base = random.uniform(branch_add_base_range[0], branch_add_base_range[1])

    change_ct_value_and_establish_clot_array(potential_sample_list, new_clot_loc_set, depth_dict, center_location_clot,
                                             partial(func_change_ct, add_base=branch_add_base, power=branch_power))

    return sample_sequence_copy


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


def change_ct_value_and_establish_clot_array(potential_sample_list, loc_clot_set, depth_dict, center_location_clot,
                                             func_change_ct):
    """

    :param potential_sample_list: a sample from the sequence
    :param loc_clot_set: return from function "get_new_loc_set"
    :param depth_dict
    :param center_location_clot
    :param func_change_ct:
    :return: None
    """
    assert len(potential_sample_list) > 0
    cub_shape = np.shape(potential_sample_list[0]['ct_data'])
    assert cub_shape[0] % 2 == 1 and cub_shape[1] % 2 and cub_shape[2] % 2

    x_radius = int(cub_shape[0] / 2)
    y_radius = int(cub_shape[1] / 2)
    z_radius = int(cub_shape[2] / 2)

    clot_center_x, clot_center_y, clot_center_z = center_location_clot

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
                local_loc = (location[0] - x_c, location[1] - y_c, location[2] - z_c)
                #if blood_vessel_depth_array[local_loc] < 0.5:
                #    continue

                relative_loc = (location[0] - clot_center_x, location[1] - clot_center_y, location[2] - clot_center_z)
                branch_level = depth_dict[relative_loc]

                add_value = func_change_ct(branch_level)

                ct_cube[local_loc] = ct_cube[local_loc] + add_value
                clot_array[local_loc] = add_value

            sample_input['clot_array'] = clot_array

    for sample in potential_sample_list:
        process_one_sample(sample)


def func_change_ct_1(branch_level, add_base, power):
    return 0.1 * (branch_level + add_base) ** power


def func_change_ct_test(branch_level, add_base, power):
    return 1


def check_correctness_1(clot_sample_dict):
    bounding_box = clot_sample_dict["range_clot"]
    loc_set = clot_sample_dict["loc_clot_set"]
    print("length loc set:", len(loc_set))
    depth_dict = clot_sample_dict["clot_depth_dict"]

    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box

    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')

    for loc in loc_set:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = depth_dict[loc]

    new_array = np.array(temp_array > 0.5, 'float32')
    import analysis.connect_region_detect as connect_region_detect
    import analysis.center_line_and_depth_3D as get_depth
    import visualization.visualize_3d.visualize_stl as stl

    new_array = connect_region_detect.convert_to_simply_connected_old(new_array, dimension=3, add_outer_layer=2,
                                                                      return_array_dtype='float32')

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

    new_sample_sequence = apply_clot_on_sample_sequence(sample_sequence, clot_sample_dict, func_change_ct_test)

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


if __name__ == '__main__':

    import Tool_Functions.Functions as Functions
    from med_transformer.image_transformer.transformer_for_PE.generate_clot_samlpe_dict import get_clot_sample_dict_from_loc_list

    temp_clot = np.zeros([80, 80, 80], 'float32')

    temp_clot[10: 70, 10: 70, 10: 70] = 1

    clot_loc_list = Functions.get_location_list(np.where(temp_clot > 0.5))

    temp_clot_sample = get_clot_sample_dict_from_loc_list(clot_loc_list)

    sample_sequence_test = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise_new/four_center_data/v1/Scanner-A-A10.pickle'
    )

    check_correctness_3(temp_clot_sample, sample_sequence_test)

    exit()

    list_sample_dict_reduced = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/temp/Scanner-A-A12.pickle')

    for i in range(0, len(list_sample_dict_reduced)):
        print(i)
        print(list_sample_dict_reduced[i]['range_clot'])



    exit()
