"""

see function "random_select_clot_sample_dict" for select clot seed
see function "apply_multiple_clot" for apply clot

clot_sample_dict: a dict, with key "loc_depth_set" and "range_clot"
clot_sample_dict["loc_clot_set"] = {(x, y, z), }
clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, ..., 'max_depth': max_depth}  here b is the clot depth
the mass center for the location x, y, z is (0, 0, 0)
clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations


sample_sequence_copy: a list, each item is a dict:
{'ct_data': ct_cube in np_array, 'penalty_weight': None, 'location_offset': central_location_offset, e.g., (23, 12, 9),
'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'clot_depth': clot_depth,
'clot_array': None}


# add value on simulated clot region:
func_increase_ct = partial(func_change_ct, add_base=depth_add_base, power=depth_power)
add_value = func_increase_ct(clot_depth) / max_possible_increase * increase_factor * random.uniform(
                    voxel_variance[0], voxel_variance[1]) * random.uniform(clot_depth / max_clot_depth, 1)
ct_cube[local_loc] = ct_cube[local_loc] + add_value

"""

import numpy as np
import copy
import random
from functools import partial
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.utlis.data_augmentation import random_flip_rotate_swap_sample, get_labels


def apply_clot_on_sample_sequence_parallel(sample, clot_sample_dict, func_change_ct,
                                           power_range=(-0.3, 0.6), add_base_range=(0, 3),
                                           value_increase=(0.01, 0.02), voxel_variance=(0.5, 1), send_end=None,
                                           augment=True, augment_labels=None, lesion_id=1, deep_copy=True,
                                           return_sample=False, ratio_clot_mass_center_on_center_line=0.5):
    """

    :param ratio_clot_mass_center_on_center_line:
    :param return_sample: True for return sample, False for return sample sequence
    :param deep_copy:
    :param lesion_id: int, start from 1, record the lesion id
            sequence_item["clot_id_array"][loc] = lesion_id

    :param augment_labels:
    :param augment: apply data augmentation
    :param send_end:
    :param value_increase:
    :param power_range:
    :param add_base_range:
    :param sample: {"center_line_loc_array": , "sample_sequence": , "additional_info": ,}
    :param clot_sample_dict:
    :param func_change_ct: func_increase_ct(clot_depth, add_base, power),
    return a float, will be added to the ct_cube at the location
    :param voxel_variance:
    :return: sample_sequence added with blood clot
    """
    assert type(sample) is dict
    if not augment:
        if deep_copy:
            sample_copy = copy.deepcopy(sample)
        else:
            sample_copy = sample
    else:
        sample_copy = random_flip_rotate_swap_sample(sample, labels=augment_labels)  # augment will do deepcopy

    sample_sequence_copy = sample_copy["sample_sequence"]

    if "center_line_loc_array" not in sample.keys():
        ratio_clot_mass_center_on_center_line = 0

    while True:
        failure_count = 0
        # determine the mass center for the clot
        if random.uniform(0, 1) < ratio_clot_mass_center_on_center_line:
            center_line_loc_array = sample_copy["center_line_loc_array"]
            center_line_loc_list = Functions.get_location_list(center_line_loc_array)
            index_clot_center = random.randint(0, len(center_line_loc_list) - 1)
            center_location_clot = center_line_loc_list[index_clot_center]
            # this location on the center-line is selected as the mass center of clot
        else:
            index_clot_center = random.randint(0, len(sample_sequence_copy) - 1)
            center_location_clot = sample_sequence_copy[index_clot_center]["center_location"]
            # sample_sequence_copy[index_clot_center] is selected as the cube lays in the mass center of clot
            if np.sum(sample_sequence_copy[index_clot_center]['depth_cube']) == 0:  # no blood vessel in this cube
                potential_item_list = []
                continue

        potential_item_list = select_potential_cubes(sample_sequence_copy, center_location_clot, clot_sample_dict)
        # use bounding box to reduce searching range
        if len(potential_item_list) > 0:
            break
        else:
            failure_count += 1
            if failure_count > 10:
                print("wrong sample sequence or wrong clot_sample")
                raise ValueError

    assert len(potential_item_list) > 0

    new_clot_loc_set = get_new_loc_set(center_location_clot, clot_sample_dict)
    # update the location_set of where has clot

    depth_power = random.uniform(power_range[0], power_range[1])
    depth_add_base = random.uniform(add_base_range[0], add_base_range[1])

    increase_factor = random.uniform(value_increase[0], value_increase[1])

    func_increase_ct = partial(func_change_ct, add_base=depth_add_base, power=depth_power)

    clot_depth_dict = clot_sample_dict["clot_depth_dict"]
    max_clot_depth = clot_depth_dict['max_depth'] + depth_add_base

    # print("depth_power:", depth_power, "depth_add_base:", depth_add_base, 'max_clot_depth:', max_clot_depth)

    max_possible_value = max(abs(func_increase_ct(1)), abs(func_increase_ct(max_clot_depth)))  # use this to normalize

    num_clot_voxels = change_ct_value_and_establish_clot_array(potential_item_list, new_clot_loc_set, clot_depth_dict,
                                                               center_location_clot, func_increase_ct,
                                                               max_possible_value, increase_factor, voxel_variance,
                                                               max_clot_depth, lesion_id)

    if not return_sample:
        # model only operate on sample_sequence
        if send_end is not None:
            send_end.send((sample_sequence_copy, num_clot_voxels))
        else:
            return sample_sequence_copy, num_clot_voxels
    else:
        sample_copy["sample_sequence"] = sample_sequence_copy
        if send_end is not None:
            send_end.send((sample_copy, num_clot_voxels))
        else:
            return sample_copy, num_clot_voxels


def random_select_clot_sample_dict(list_clot_sample_dict, target_volume=None, max_trial=np.inf, raise_error=True):
    """

    :param list_clot_sample_dict:
    :param target_volume: the range of raw volume of the clot, like (2000, 20000), like (1000, np.inf), like (0, 1000)
    :param max_trial:
    :param raise_error
    :return: clot_sample_dict of the given volume range
    """
    total_clots = len(list_clot_sample_dict)
    if target_volume is None:
        return list_clot_sample_dict[random.randint(0, total_clots - 1)]
    assert target_volume[1] > 300 and target_volume[0] < 30000 and target_volume[1] > target_volume[0]

    index_search_list = list(np.arange(0, total_clots))
    random.shuffle(index_search_list)

    pointer = 0
    candidate_clot_sample = list_clot_sample_dict[index_search_list[pointer]]
    clot_volume = len(candidate_clot_sample["loc_clot_set"])
    while not target_volume[0] <= clot_volume <= target_volume[1]:
        pointer += 1
        if pointer > max_trial:
            print("exceed max trial!")
            if raise_error:
                raise ValueError
            else:
                return None
        if pointer >= total_clots:
            print("no applicable clot!")
            if raise_error:
                raise ValueError
            else:
                return None
        candidate_clot_sample = list_clot_sample_dict[index_search_list[pointer]]
        clot_volume = len(candidate_clot_sample["loc_clot_set"])
    return candidate_clot_sample


def apply_clot_on_sample_until_satisfy(sample, clot_sample_dict, func_apply_clot, lesion_id=1, min_volume=250,
                                       max_trial=10, remain_unqualified=True):
    """

    :param lesion_id:
    :param remain_unqualified: if exceed max_trial, still not reach min_volume, return the max_volume trial
    :param sample:
    :param clot_sample_dict:
    :param func_apply_clot: sample_sequence, num_clot_voxels = func_apply_clot(sample, clot_sample_dict, lesion_id)
    :param min_volume: if the lesion is on boundary, it may too little
    :param max_trial:
    :return: sample_with_clot, num_clot_voxels
    """
    sample_with_clot, num_clot_voxels = func_apply_clot(sample, clot_sample_dict, lesion_id=lesion_id)
    trial_count = 0
    while num_clot_voxels < min_volume and trial_count < max_trial:
        sample_with_clot_temp, num_clot_voxels_temp = func_apply_clot(sample, clot_sample_dict, lesion_id=lesion_id)
        if num_clot_voxels_temp > num_clot_voxels:
            num_clot_voxels = num_clot_voxels_temp
            sample_with_clot = sample_with_clot_temp
        trial_count += 1

    if remain_unqualified:
        return sample_with_clot, num_clot_voxels

    if num_clot_voxels >= min_volume:
        return sample_with_clot, num_clot_voxels

    return None, None


def apply_multiple_clot(sample, list_clot_sample_dict, func_apply_clot, min_volume=250, max_trial=10,
                        remain_unqualified=True, send_end=None, augment=True, augment_labels=None,
                        return_id_volume_dict=False):
    """

    :param return_id_volume_dict:
    :param sample:
    :param list_clot_sample_dict:
    :param func_apply_clot:
    :param min_volume:
    :param max_trial:
    :param remain_unqualified:
    :param send_end:
    :param augment:
    :param augment_labels:
    :return: sample_sequence_with_clot or (sample_sequence_with_clots, {1: volume_of_clot_1, 2: volume_of_clot_2, ...})
    """
    assert type(sample) is dict
    if not augment:
        sample_copy = copy.deepcopy(sample)
    else:
        if augment_labels is None:
            augment_labels = get_labels(swap_axis=False)  # rotate and flip is enough
        else:
            assert augment_labels[2] == 0  # no swap
        sample_copy = random_flip_rotate_swap_sample(sample, labels=augment_labels)  # augment will do deepcopy

    lesion_id_volume_dict = {}

    lesion_id = 1
    for clot_sample_dict in list_clot_sample_dict:
        sample_copy, num_clot_voxels = apply_clot_on_sample_until_satisfy(
            sample_copy, clot_sample_dict, func_apply_clot, lesion_id, min_volume, max_trial, remain_unqualified)
        lesion_id_volume_dict[lesion_id] = num_clot_voxels
        lesion_id += 1

    # model only operate on sample_sequence
    sample_sequence_with_clot = sample_copy["sample_sequence"]
    assign_lesion_volume_array(sample_sequence_with_clot, lesion_id_volume_dict, min_volume=min_volume)
    if send_end is not None:
        if return_id_volume_dict:
            send_end.send((sample_sequence_with_clot, lesion_id_volume_dict))
        send_end.send(sample_sequence_with_clot)
    else:
        if return_id_volume_dict:
            return sample_sequence_with_clot, lesion_id_volume_dict
        return sample_sequence_with_clot


def assign_lesion_volume_array(sample_sequence_with_clot, lesion_id_volume_dict, min_volume=None):
    """
    create key "lesion_volume_array" for each item in sample_sequence_with_clot
    for this array, 0 means not clot, >0 means the volume for the clot

    :param min_volume:
    :param sample_sequence_with_clot:
    :param lesion_id_volume_dict:
    :return: sample_sequence_with_clot
    """
    cube_shape = np.shape(sample_sequence_with_clot[0]['ct_data'])
    if min_volume is None:
        min_volume = 0
    for item in sample_sequence_with_clot:
        if 'clot_id_array' not in list(item.keys()):
            item['clot_volume_array'] = None
            continue
        clot_volume_array = np.zeros(cube_shape, 'float32')
        clot_id_array = item['clot_id_array']
        loc_list_clot = Functions.get_location_list(np.where(clot_id_array > 0.5))
        for loc in loc_list_clot:
            clot_volume_array[loc] = max(lesion_id_volume_dict[clot_id_array[loc]], min_volume)
        item['clot_volume_array'] = clot_volume_array


def get_new_loc_set(center_location_clot, clot_sample_dict):
    initial_loc_set = clot_sample_dict["loc_clot_set"]

    x_c, y_c, z_c = center_location_clot

    new_loc_set = set()

    for location in initial_loc_set:
        new_loc_set.add((location[0] + x_c, location[1] + y_c, location[2] + z_c))

    return new_loc_set


def select_potential_cubes(sample_sequence_copy, center_location_clot, clot_sample_dict):
    """

    :param sample_sequence_copy:
    :param center_location_clot: the mass center of the clot
    :param clot_sample_dict:
    :return:
    """

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


def change_ct_value_and_establish_clot_array(potential_item_list, loc_clot_set, clot_depth_dict, center_location_clot,
                                             func_increase_ct, max_possible_increase, increase_factor, voxel_variance,
                                             max_clot_depth, lesion_id):
    """

    :param lesion_id:
    :param max_clot_depth: max_clot_depth. small depth means lower increase value
    :param potential_item_list: a sample from the sequence
    :param loc_clot_set: return from function "get_new_loc_set"
    :param clot_depth_dict
    :param center_location_clot
    :param func_increase_ct:
    :param max_possible_increase: the max_possible_increase for func_increase_ct
    :param increase_factor
    :param voxel_variance
    :return: num_voxel_changed  # the number of clot voxels
    """
    assert len(potential_item_list) > 0
    assert lesion_id >= 1
    cub_shape = np.shape(potential_item_list[0]['ct_data'])
    assert cub_shape[0] % 2 == 1 and cub_shape[1] % 2 and cub_shape[2] % 2

    x_radius = int(cub_shape[0] / 2)
    y_radius = int(cub_shape[1] / 2)
    z_radius = int(cub_shape[2] / 2)

    clot_center_x, clot_center_y, clot_center_z = center_location_clot

    num_voxel_changed = np.array([0, ], 'int32')  # the number of clot voxels

    def process_one_item(item_input):
        ct_cube = item_input['ct_data']
        depth_cube = item_input['depth_cube']
        loc_set_sample = set()
        x_c, y_c, z_c = item_input['center_location']
        for x in range(-x_radius, x_radius + 1):
            for y in range(-y_radius, y_radius + 1):
                for z in range(-z_radius, z_radius + 1):
                    loc_set_sample.add((x_c + x, y_c + y, z_c + z))

        intersection_loc_set = loc_set_sample & loc_clot_set  # the absolute locations

        if len(intersection_loc_set) > 0:
            clot_array = np.zeros(cub_shape, 'float32')
            if "clot_id_array" not in list(item_input.keys()):
                clot_id_array = np.zeros(cub_shape, 'float32')
            else:
                clot_id_array = item_input["clot_id_array"]

            for location in intersection_loc_set:
                local_loc = (location[0] - x_c + x_radius, location[1] - y_c + y_radius, location[2] - z_c + z_radius)
                if depth_cube[local_loc] < 0.5:  # not blood region
                    continue

                relative_loc = (location[0] - clot_center_x, location[1] - clot_center_y, location[2] - clot_center_z)
                clot_depth = clot_depth_dict[relative_loc]

                add_value = func_increase_ct(clot_depth) / max_possible_increase * increase_factor * random.uniform(
                    voxel_variance[0], voxel_variance[1]) * random.uniform(clot_depth / max_clot_depth, 1)

                ct_cube[local_loc] = ct_cube[local_loc] + add_value
                clot_array[local_loc] = add_value
                clot_id_array[local_loc] = lesion_id
                num_voxel_changed[0] = num_voxel_changed[0] + 1

            if 'clot_array' not in item_input.keys():
                item_input['clot_array'] = clot_array
            else:
                if item_input['clot_array'] is None:
                    item_input['clot_array'] = clot_array
                else:
                    item_input['clot_array'] = item_input['clot_array'] + clot_array

            item_input["clot_id_array"] = clot_id_array

    for item in potential_item_list:
        process_one_item(item)

    return num_voxel_changed[0]


def func_change_ct_default(clot_depth, add_base, power):
    return (clot_depth + add_base) ** power


def func_change_ct_test(clot_depth, add_base, power, test=True):
    if test:
        return 1
    return func_change_ct_default(clot_depth, add_base, power)


def apply_clot_on_sample(sample, list_clot_sample_dict, func_change_ct=func_change_ct_default, power_range=(-0.3, 0.6),
                         add_base_range=(0, 3), value_increase=(0.01, 0.02), voxel_variance=(0.99, 1),
                         min_volume=250, max_trial=10, augment=True, visualize=False,
                         ratio_clot_mass_center_on_center_line=0.5):
    """

    :param ratio_clot_mass_center_on_center_line:
    :param augment:
    :param sample: {"center_line_loc_array": , "sample_sequence": , "additional_info": ,}
    :param list_clot_sample_dict:
    :param func_change_ct: func_change_ct(clot_depth, add_base, power)
    :param power_range:
    :param add_base_range:
    :param value_increase:
    :param voxel_variance:
    :param min_volume:
    :param max_trial:
    :param visualize:
    :return: sample_sequence added with blood clots (with type sample["sample_sequence"])
    """
    if augment:
        augment_labels = get_labels(swap_axis=False)
        if visualize:
            print("augment labels:", augment_labels)
        sample_copy = random_flip_rotate_swap_sample(sample, labels=augment_labels)  # augment will do deepcopy
    else:
        sample_copy = copy.deepcopy(sample)

    if len(list_clot_sample_dict) == 0:
        if visualize:
            print("no clot to simulate")
        return sample_copy["sample_sequence"]

    func_apply_clot = partial(
        apply_clot_on_sample_sequence_parallel, func_change_ct=func_change_ct, power_range=power_range,
        add_base_range=add_base_range, value_increase=value_increase, voxel_variance=voxel_variance,
        augment=False, send_end=None,
        return_sample=True, deep_copy=False, ratio_clot_mass_center_on_center_line=ratio_clot_mass_center_on_center_line
    )

    sample_sequence_with_clot, lesion_id_volume_dict = apply_multiple_clot(
        sample_copy, list_clot_sample_dict, func_apply_clot, return_id_volume_dict=True,
        min_volume=min_volume, max_trial=max_trial, remain_unqualified=True)

    if visualize:
        print("lesion_id_volume_dict:\n", lesion_id_volume_dict)

    return sample_sequence_with_clot


def visualize_clots_on_sample_sequence(sample_sequence_with_clot, high_resolution=True, clip_window=False):
    import pulmonary_embolism_final.utlis.ct_sample_sequence_converter as converter
    import visualization.visualize_3d.visualize_stl as stl

    for item in sample_sequence_with_clot:
        if 'clot_array' in item.keys():
            print("keys for item:", list(item.keys()))
            break

    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
    else:
        absolute_cube_length = (4, 4, 5)

    clot_array = converter.reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length, key='clot_array')
    clot_mask = np.array(clot_array > 0, 'float32') + np.array(clot_array < 0, 'float32')

    # clot_volume_array = converter.reconstruct_rescaled_ct_from_sample_sequence(
    #     sample_sequence_with_clot, absolute_cube_length, key='clot_volume_array')

    blood_depth = converter.reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length, key='depth_cube')

    blood_region = np.array(blood_depth > 0, 'float32')

    print("blood vessel mask")
    stl.visualize_numpy_as_stl(blood_region)

    ct_array = converter.reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length, key='ct_data')

    if clip_window:
        ct_array = np.clip(ct_array, Functions.change_to_rescaled(-400), Functions.change_to_rescaled(400))

    loc_list_z_clot = list(set(np.where(clot_mask > 0)[2]))
    loc_list_z_clot.sort()

    for z in loc_list_z_clot[::5]:
        print(z)
        Functions.merge_image_with_mask(ct_array[:, :, z], clot_mask[:, :, z], show=True)


def check_lesion_apply():
    reduced_clot_sample_list = Functions.pickle_load_object(
        '/data_disk/pulmonary_embolism/simulated_lesions/clot_sample_list_reduced/merged/clot_sample_list_2%.pickle')
    print(len(reduced_clot_sample_list))

    temp_clot_sample_list = []
    for i in range(5):
        print(i)
        temp_clot_sample_list.append(random_select_clot_sample_dict(reduced_clot_sample_list, (500, np.inf)))

    sample_test = Functions.pickle_load_object(
        # '/data_disk/pulmonary_embolism/training_dataset_new/combine_ready_denoise/disk4-11_2020-04-27.pickle'
        '/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_high_recall/'
        'low_resolution/pe_ready_denoise/patient-id-135.pickle'
    )

    sample_sequence_with_clot = apply_clot_on_sample(sample_test, temp_clot_sample_list,
                                                     value_increase=(0.2, 1.0), visualize=True, min_volume=50,
                                                     augment=False)

    visualize_clots_on_sample_sequence(sample_sequence_with_clot, high_resolution=False)


if __name__ == '__main__':
    check_lesion_apply()
