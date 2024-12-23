"""
clot_sample_dict: a dict, with key "loc_depth_set" and "range_clot"
clot_sample_dict["loc_clot_set"] = {(x, y, z), }, with mass center at (0, 0, 0)
clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, }  here b is the branching level
the mass center for the location x, y, z is (0, 0, 0)
clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations

"""

import numpy as np
import Tool_Functions.Functions as Functions
import analysis.connect_region_detect as connect_region_detect
import analysis.center_line_and_depth_3D as get_depth


def remove_holes(loc_list):
    new_loc_list, mass_center, bounding_box = get_bounding_box_and_mass_center(loc_list, show=False)
    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box
    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')
    for loc in new_loc_list:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = 1
    temp_array = connect_region_detect.convert_to_simply_connected_old(temp_array, dimension=3, add_outer_layer=5,
                                                                       return_array_dtype='float32')

    loc_list_clot = Functions.get_location_list(np.where(temp_array > 0.5))
    return loc_list_clot


def get_clot_sample_dict_from_loc_list(loc_list_clot, remove_hole=True):

    if remove_hole:
        loc_list_clot = remove_holes(loc_list_clot)

    new_loc_list, mass_center, bounding_box = get_bounding_box_and_mass_center(loc_list_clot, show=False)

    clot_sample_dict = {"loc_clot_set": set(new_loc_list), "clot_depth_dict": dict(), "range_clot": None}

    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box

    clot_sample_dict["range_clot"] = ((int(x_min), int(x_max)), (int(y_min), int(y_max)), (int(z_min), int(z_max)))

    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')

    for loc in new_loc_list:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = 1

    depth_array = get_depth.get_surface_distance(temp_array)

    for loc in new_loc_list:
        clot_sample_dict["clot_depth_dict"][loc] = \
            depth_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))]

    return clot_sample_dict


def get_bounding_box_and_mass_center(loc_list, show=True):
    loc_array = list(Functions.get_location_array(loc_list))

    x_c = np.median(loc_array[0])
    y_c = np.median(loc_array[1])
    z_c = np.median(loc_array[2])

    loc_array[0] = loc_array[0] - x_c
    loc_array[1] = loc_array[1] - y_c
    loc_array[2] = loc_array[2] - z_c

    new_loc_list = Functions.get_location_list(loc_array)

    x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
    y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])
    z_min, z_max = np.min(loc_array[2]), np.max(loc_array[2])

    mass_center = (x_c, y_c, z_c)
    bounding_box = ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    if show:
        print("mass center:", mass_center)
        print("bounding_box:", bounding_box)

    return new_loc_list, mass_center, bounding_box


def refine_simulated_clot_seed(semantic_id_loc_dict, show=True):
    """

    :param semantic_id_loc_dict: {semantic_id: id_loc_list},
    see analysis.connect_region_detect.get_connected_regions_discrete
    :param show
    :return: refined clot, is a id_loc_list_sorted that merged all_file semantic_id
    """

    refined_id_loc_dict = {}
    qualified_region_id = 1

    for semantic_id, sub_id_loc_dict in semantic_id_loc_dict.items():
        print(semantic_id, len(semantic_id_loc_dict) + 1)
        for region_id in range(1, len(sub_id_loc_dict) + 1):
            loc_list = sub_id_loc_dict[region_id]
            new_loc_list, mass_center, bounding_box = get_bounding_box_and_mass_center(loc_list, show=False)
            ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box
            if x_max - x_min > 60 or y_max - y_min > 60 or z_max - z_min > 60:
                continue
            refined_id_loc_dict[qualified_region_id] = new_loc_list
            qualified_region_id += 1

    refined_id_loc_dict_sorted, id_volume_dict = connect_region_detect.sort_on_id_loc_dict(refined_id_loc_dict)

    if show:
        print("there are", len(id_volume_dict), "regions")
        print("top 100:")
        temp_dict = {}
        for key in range(1, min(len(id_volume_dict), 100) + 1):
            temp_dict[key] = id_volume_dict[key]
        print(temp_dict)

    return refined_id_loc_dict_sorted


def form_sample_dict_list(id_loc_dict, sample_dict_list_save_path):
    print("form final sample dict list")
    sample_dict_list = []
    for clot_key, loc_list_clot in id_loc_dict.items():
        if clot_key % 10 == 0:
            print(clot_key, '/', len(id_loc_dict))
        clot_sample_dict = get_clot_sample_dict_from_loc_list(loc_list_clot)
        sample_dict_list.append(clot_sample_dict)
    Functions.pickle_save_object(sample_dict_list_save_path, sample_dict_list)


def form_sample_dict_list_from_ct(ct_denoise, lung_mask, save_path):
    import time
    bounding_lung = Functions.get_bounding_box(lung_mask)

    ct_denoise = np.clip(ct_denoise * 1600 - 600 + 1000, 0, 2000) / 5
    ct_denoise = np.array(ct_denoise, 'int16')
    # have to cast to int for "connect_region_detect.get_connected_regions_discrete"

    sub_array = ct_denoise[bounding_lung[0][0]: bounding_lung[0][1], bounding_lung[1][0]: bounding_lung[1][1],
                           bounding_lung[2][0]: bounding_lung[2][1]]
    print("shape search array", np.shape(sub_array))

    print("getting semantic_loc dict")

    time_start = time.time()
    id_sorted_loc_dict = connect_region_detect.get_connected_regions_discrete(sub_array, strict=False, show=True)
    time_end = time.time()

    print(time_start - time_end)

    len_id = len(id_sorted_loc_dict)

    semantic_id_loc_dict = {}

    for index in range(len_id):
        print(index)
        loc_dict = id_sorted_loc_dict[index]
        sub_dict = {}
        sub_dict_loc_list = {}
        for key, loc_list in loc_dict.items():
            if len(loc_list) < 750:
                break
            sub_dict[key] = len(loc_list)
            sub_dict_loc_list[key] = loc_list

        if len(sub_dict) > 0:
            print("id_volume dict for semantic", index, '\n', sub_dict)
            semantic_id_loc_dict[index] = sub_dict_loc_list

    refined_id_loc_dict_sorted = refine_simulated_clot_seed(semantic_id_loc_dict)

    form_sample_dict_list(refined_id_loc_dict_sorted, save_path)


if __name__ == '__main__':

    import os
    save_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/list-clot_sample_dict/'

    fn_list = os.listdir('/data_disk/rescaled_ct_and_semantics/denoise_new_ct_float16/healthy_people/four_center_data/')

    for fn in fn_list[::10]:
        test_array = np.load(
            '/data_disk/rescaled_ct_and_semantics/denoise_new_ct_float16/healthy_people/four_center_data/' + fn)[
            'array']

        mask_lung = np.load(
            '/data_disk/rescaled_ct_and_semantics/semantics/healthy_people/four_center_data/lung_mask/' + fn)[
            'array']

        form_sample_dict_list_from_ct(test_array, mask_lung,
                                      save_top_dict + fn[:-4] + '.pickle')

        exit()

