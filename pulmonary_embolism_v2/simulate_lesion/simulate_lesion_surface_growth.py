"""
many lesion seed with 50-100 voxels:
lesion seed is a loc list, [(x, y, z), ], with mass center very close to (0, 0, 0)


"""
import random
import numpy as np
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.visualize_stl as stl
import analysis.connect_region_detect as connect_region_detect


def remove_holes(loc_list_or_loc_array, add_surface=0, max_hole_radius=2):

    if type(loc_list_or_loc_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_loc_array)
    else:
        loc_array = loc_list_or_loc_array

    if len(loc_array[0]) == 0:
        return loc_array

    pad = 5 + 2 * add_surface
    assert len(loc_array) == 2 or len(loc_array) == 3

    if len(loc_array) == 3:
        x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
        y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])
        z_min, z_max = np.min(loc_array[2]), np.max(loc_array[2])

        temp_array = np.zeros([int(x_max - x_min + pad), int(y_max - y_min + pad), int(z_max - z_min + pad)], 'float32')

        new_loc_array = (loc_array[0] - x_min + int(pad / 2),
                         loc_array[1] - y_min + int(pad / 2),
                         loc_array[2] - z_min + int(pad / 2))

        temp_array[new_loc_array] = 1
        temp_array = connect_region_detect.convert_to_simply_connected(temp_array, max_hole_radius, add_surface)
        new_loc_array = np.where(temp_array > 0.5)

        loc_array_no_hole = (new_loc_array[0] + x_min - int(pad / 2),
                             new_loc_array[1] + y_min - int(pad / 2),
                             new_loc_array[2] + z_min - int(pad / 2))

        if type(loc_list_or_loc_array) is list:
            return Functions.get_location_list(loc_array_no_hole)
        return loc_array_no_hole
    else:
        x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
        y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])

        temp_array = np.zeros([int(x_max - x_min + pad), int(y_max - y_min + pad)], 'float32')

        new_loc_array = (loc_array[0] - x_min + int(pad / 2),
                         loc_array[1] - y_min + int(pad / 2))

        temp_array[new_loc_array] = 1
        temp_array = connect_region_detect.convert_to_simply_connected(temp_array, max_hole_radius, add_surface)
        new_loc_array = np.where(temp_array > 0.5)

        loc_array_no_hole = (new_loc_array[0] + x_min - int(pad / 2),
                             new_loc_array[1] + y_min - int(pad / 2))

        if type(loc_list_or_loc_array) is list:
            return Functions.get_location_list(loc_array_no_hole)
        return loc_array_no_hole


def random_lesion_growth(lesion_seed_list, target_volume, bias_func=None):
    """

    :param lesion_seed_list: a list of lesion_seed
    :param target_volume: terminate if lesion_volume >= target_volume
    :param bias_func: a function, input a list of locations, output one location, i.e., the point for lesion growth
    :return: the loc_list for the lesion
    """
    def default_bias_func(loc_list):
        return loc_list[random.randint(0, len(loc_list) - 1)]
    if bias_func is None:
        bias_func = default_bias_func

    show_new_add_ratio = False  # how many locations will the new lesion be used? Answer: average 0.5, std 0.25

    total_seeds = len(lesion_seed_list)
    initial_lesion = lesion_seed_list[random.randint(0, total_seeds - 1)]

    set_non_overlap_loc = set(initial_lesion)  # these locations is not overlap with existing seeds
    list_non_overlap_loc = list(initial_lesion)
    set_lesion_loc = set(initial_lesion)  # the lesion locations

    list_new_add_ratio = []

    while len(set_lesion_loc) < target_volume:
        # get new lesion_seed
        new_lesion_loc_list = lesion_seed_list[random.randint(0, total_seeds - 1)]
        new_lesion_loc_array = list(Functions.get_location_array(new_lesion_loc_list))
        # get mass center for new lesion
        center_for_new_lesion = bias_func(list_non_overlap_loc)
        # change mass center
        new_lesion_loc_array[0] = new_lesion_loc_array[0] + center_for_new_lesion[0]
        new_lesion_loc_array[1] = new_lesion_loc_array[1] + center_for_new_lesion[1]
        new_lesion_loc_array[2] = new_lesion_loc_array[2] + center_for_new_lesion[2]
        new_lesion_loc_list = Functions.get_location_list(new_lesion_loc_array)
        new_lesion_loc_set = set(new_lesion_loc_list)

        # updating...
        if show_new_add_ratio:
            list_new_add_ratio.append(1 - len(new_lesion_loc_set & set_lesion_loc) / len(new_lesion_loc_set))
        new_overlap_region = set_non_overlap_loc & new_lesion_loc_set
        set_non_overlap_loc.difference_update(new_overlap_region)
        first_time_loc = new_lesion_loc_set - set_lesion_loc  # this must be non_overlap
        set_non_overlap_loc = set_non_overlap_loc | first_time_loc
        set_lesion_loc = set_lesion_loc | first_time_loc
        list_non_overlap_loc = list(set_non_overlap_loc)

    if show_new_add_ratio:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.displot(list_new_add_ratio)
        print(len(list_new_add_ratio), np.median(list_new_add_ratio), np.average(list_new_add_ratio),
              np.std(list_new_add_ratio))
        plt.show()

    return list(set_lesion_loc)


def fractal_lesion_growth(lesion_seed_list, target_volume, scale_factor=1.5, bias_func=None, in_recursive=False):
    """

    Let the overlap ratio between lesion and new lesion as:
    len(new_lesion_loc_set & set_lesion_loc) / len(new_lesion_loc_set)
    In random growth, this ratio is of average 0.5, std 0.25
    Thus, we assume this ratio is also (or should close) of average 0.5, std 0.25 in fractal growth

    when growth to the next "level", its expectation volume increase by "scale_factor"
    level n-1 as the new lesion (small), it add to the surface of a big lesion, let expect volume for level n as V_n.
    If current volume V_(n-1), want to grow to V_n. We need a lesion with volume (scale_factor- 0.5) * V_(n-1), then add
    V_(n-1) to the surface of this lesion we have: (scale_factor- 0.5) * V_(n-1) + 0.5 * V_(n-1) = V_n

    :param lesion_seed_list: a list of lesion_seed
    :param target_volume: terminate if lesion_volume >= target_volume
    :param bias_func: a function, input a list of locations, output one location, i.e., the point for lesion growth

    :param scale_factor:
    the expectation shape for lesion is the same for these two lesions with volume V and V * scale_factor

    :param in_recursive: True for return the list_non_overlap_loc
    :return: the loc_list for the lesion
    """

    assert 1 < scale_factor

    def default_bias_func(loc_list):
        return loc_list[random.randint(0, len(loc_list) - 1)]
    if bias_func is None:
        bias_func = default_bias_func

    total_seeds = len(lesion_seed_list)
    initial_lesion = lesion_seed_list[random.randint(0, total_seeds - 1)]

    set_non_overlap_loc = set(initial_lesion)  # these locations is not overlap with existing seeds
    list_non_overlap_loc = list(initial_lesion)
    list_lesion_loc = initial_lesion
    set_lesion_loc = set(list_lesion_loc)

    if target_volume <= 100:
        return random_lesion_growth(lesion_seed_list, target_volume, bias_func)

    def grow_one_scale():
        # current volume is at V_(n-1), we need add it to a volume of (scale_factor- 0.5) * V_(n-1)
        volume_base = (scale_factor - 0.5) * len(list_lesion_loc)
        set_loc_base, set_non_overlap_base = fractal_lesion_growth(
            lesion_seed_list, volume_base, scale_factor, bias_func, True)
        list_non_overlap_base = list(set_non_overlap_base)

        # V_(n-1) is the new lesion, and will add to loc_list_base
        lesion_loc_array = list(Functions.get_location_array(list_lesion_loc))
        center_for_lesion = bias_func(list_non_overlap_base)
        lesion_loc_array[0] = lesion_loc_array[0] + center_for_lesion[0]
        lesion_loc_array[1] = lesion_loc_array[1] + center_for_lesion[1]
        lesion_loc_array[2] = lesion_loc_array[2] + center_for_lesion[2]
        new_lesion_loc_list = Functions.get_location_list(lesion_loc_array)
        new_lesion_loc_set = set(new_lesion_loc_list)

        new_overlap_region = set_non_overlap_base & new_lesion_loc_set
        set_non_overlap_base.difference_update(new_overlap_region)
        first_time_loc = new_lesion_loc_set - set_loc_base  # this must be non_overlap
        set_non_overlap_base = set_non_overlap_base | first_time_loc
        set_lesion_next_level = set_loc_base | first_time_loc
        list_non_overlap_loc = list(set_non_overlap_loc)

    if in_recursive:
        return set_lesion_loc, set_non_overlap_loc
    return list_lesion_loc


def run_func_get_lesion_seed():
    import os

    save_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/surface_growth/lesion_seed/'

    fn_list = os.listdir('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/xwzc/')

    for fn in fn_list:
        test_array = np.load(
            '/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/healthy_people/xwzc/' + fn)[
            'array']

        mask_lung = np.load(
            '/data_disk/rescaled_ct_and_semantics/semantics/healthy_people/xwzc/lung_mask/' + fn)[
            'array']

        get_lesion_seed(test_array, mask_lung, save_top_dict + fn[:-4] + '.pickle')


def get_lesion_seed(rescaled_ct, region_of_interest, pickle_save_path, seed_voxel_count_range=(50, 100),
                    change_to_hu=True, strict=False):
    """

    :param rescaled_ct:
    :param region_of_interest:
    :param pickle_save_path:
    save path for the seed_list: an id_loc_dict, each item is the location_list of a tiny connected component, and the
    mass center for these locations is set to (0, 0, 0)
    :param seed_voxel_count_range: the length range of the item of the seed_list
    :param change_to_hu
    :param strict: the criteria for connected component
    :return: None
    """
    assert seed_voxel_count_range[1] > seed_voxel_count_range[0]
    import time
    if change_to_hu:
        ct_data = Functions.change_to_HU(rescaled_ct) / 5
    else:
        ct_data = rescaled_ct
    if region_of_interest is None:
        bounding_box = [[0, np.shape(ct_data)[0]], [0, np.shape(ct_data)[1]], [0, np.shape(ct_data)[2]]]
    else:
        bounding_box = Functions.get_bounding_box(region_of_interest)
        ct_data = ct_data * region_of_interest

    ct_data = np.array(ct_data, 'int16')
    # have to cast to int for "connect_region_detect.get_connected_regions_discrete"

    sub_array = ct_data[bounding_box[0][0]: bounding_box[0][1], bounding_box[1][0]: bounding_box[1][1],
                        bounding_box[2][0]: bounding_box[2][1]]

    print("shape search array", np.shape(sub_array))

    print("getting semantic_loc dict")

    time_start = time.time()
    id_sorted_loc_dict = connect_region_detect.get_connected_regions_discrete(sub_array, strict=strict, show=True)
    time_end = time.time()

    print("actual time spent", time_end - time_start)

    key_list = list(id_sorted_loc_dict.keys())

    semantic_id_loc_dict = {}

    for index in key_list:
        print(index)
        loc_dict = id_sorted_loc_dict[index]
        sub_dict = {}
        sub_dict_loc_list = {}
        for key, loc_list in loc_dict.items():
            if len(loc_list) < seed_voxel_count_range[0]:
                print(len(loc_list))
                break
            if len(loc_list) > seed_voxel_count_range[1]:
                continue
            sub_dict[key] = len(loc_list)
            sub_dict_loc_list[key] = loc_list

        if len(sub_dict) > 0:
            print("id_volume dict for semantic", index, '\n', sub_dict)
            semantic_id_loc_dict[index] = sub_dict_loc_list

    refined_id_loc_list = refine_simulated_clot_seed(semantic_id_loc_dict)

    Functions.pickle_save_object(pickle_save_path, refined_id_loc_list)


def get_bounding_box_zero_mass_center(loc_list, show=True):
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

    mass_center_original = (x_c, y_c, z_c)
    bounding_box = ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    if show:
        print("mass center:", mass_center_original)
        print("bounding_box:", bounding_box)

    return new_loc_list, mass_center_original, bounding_box


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
        key_list_sub_id_loc_dict = list(sub_id_loc_dict.keys())
        for region_id in key_list_sub_id_loc_dict:
            loc_list = sub_id_loc_dict[region_id]
            new_loc_list, mass_center_original, bounding_box = get_bounding_box_zero_mass_center(loc_list, show=False)
            ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box

            x_differ, y_differ, z_differ = x_max - x_min, y_max - y_min, z_max - z_min
            differ_array = [x_differ, y_differ, z_differ]
            if max(differ_array) / (min(differ_array) + 1) > 3:
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


def visualize_simulated_clot(loc_list):
    loc_array = Functions.get_location_array(loc_list)
    x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
    y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])
    z_min, z_max = np.min(loc_array[2]), np.max(loc_array[2])

    temp_array = np.zeros([int(x_max - x_min + 4), int(y_max - y_min + 4), int(z_max - z_min + 4)], 'float32')

    for loc in loc_list:
        temp_array[(int(loc[0] - x_min + 2), int(loc[1] - y_min + 2), int(loc[2] - z_min + 2))] = 1

    new_array = np.array(temp_array > 0.5, 'float32')
    import analysis.center_line_and_depth_3D as get_depth

    stl.visualize_numpy_as_stl(new_array)

    new_depth_array = get_depth.get_surface_distance(new_array)

    print("max encoding_depth", np.max(new_depth_array))

    Functions.image_show(new_depth_array[:, :, int((z_max - z_min + 4) / 2)])


def save_list_of_lesion_array(save_path, volume_base=5000, num_lesions=1000):
    """

    :param num_lesions: length for the list_lesion_loc_array
    :param save_path: path for saving the list_lesion_loc_array
    :param volume_base: lesion will with volume: (volume_base, 10 * volume_base)
    :return:
    """
    import os
    import analysis.point_cloud as point_cloud
    top_dict_seed = '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/surface_growth/lesion_seed/'
    list_seed_dict = os.listdir(top_dict_seed)
    seed_list = []
    for seed_file in list_seed_dict[10: 20]:
        seed_dict = Functions.pickle_load_object(top_dict_seed + seed_file)
        for key, seed in seed_dict.items():
            seed_list.append(seed)

    print(len(seed_list), "number of seed")

    list_of_lesion_array = []

    for count in range(num_lesions):
        if count % 10 == 0:
            print(count, '/', num_lesions)
        volume = volume_base * 10 ** random.uniform(0, 1)
        lesion_loc_list = random_lesion_growth(seed_list, int(volume))
        lesion_loc_array = Functions.get_location_array(lesion_loc_list)
        list_of_lesion_array.append(point_cloud.set_mass_center(lesion_loc_array, (0, 0, 0), show=False))

    Functions.pickle_save_object(save_path, list_of_lesion_array)


def get_different_difficult_level(lesion_list, strict=True, save_path=None):
    import analysis.get_surface_rim_adjacent_mean as get_surface
    import analysis.point_cloud as point_cloud
    new_lesion_list = []
    for index, lesion_loc_array in enumerate(lesion_list):
        if index % 10 == 0:
            print(index)
        lesion_array = point_cloud.point_cloud_to_numpy_array(lesion_loc_array, None, pad=4)

        surface = get_surface.get_surface(lesion_array, outer=True, strict=strict)
        new_lesion_array = lesion_array + surface
        new_lesion_array = new_lesion_array - get_surface.get_surface(new_lesion_array, outer=False, strict=strict)
        if strict is False:
            new_lesion_array = new_lesion_array - get_surface.get_surface(new_lesion_array, outer=False, strict=True)

        loc_array = point_cloud.set_mass_center(np.where(new_lesion_array > 0.5), show=False, int_loc_arrays=True)
        new_lesion_list.append(loc_array)

    if save_path is None:
        if strict:
            save_path = '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/simulated_results/lesion/' \
                        'list-of-loc-array_surface-growth_volume_5000-50000_lv1.pickle'
        else:
            save_path = '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/simulated_results/lesion/' \
                        'list-of-loc-array_surface-growth_volume_5000-50000_lv2.pickle'
    Functions.pickle_save_object(save_path, new_lesion_list)


if __name__ == '__main__':

    surface_growth_lv_0 = Functions.pickle_load_object('/data_disk/artery_vein_project/extract_blood_region/'
                                                       'lesion_simulation/'
                                                       'list-of-loc-array_surface-growth_volume_500-5000_lv0.pickle')

    get_different_difficult_level(surface_growth_lv_0, strict=True,
                                  save_path='/data_disk/artery_vein_project/extract_blood_region/lesion_simulation/'
                                            'list-of-loc-array_surface-growth_volume_500-5000_lv1.pickle')

    exit()

    save_list_of_lesion_array('/data_disk/artery_vein_project/extract_blood_region/lesion_simulation/'
                              'list-of-loc-array_surface-growth_volume_500-5000_lv0.pickle', 500)
    exit()



    test_image = np.zeros([100, 100])
    test_image[20:40, 30:50] = 1
    test_image[30, 35] = 0
    Functions.image_show(test_image)

    test_image_2 = np.zeros([100, 100])
    test_image_2[remove_holes(np.where(test_image > 0.5))] = 1
    Functions.image_show(test_image_2)

    exit()
    seed_dict_test = Functions.pickle_load_object('/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/'
                                                  'surface_growth/lesion_seed/xwzc000098.pickle')
    seed_list_test = []
    for key_test, seed_test in seed_dict_test.items():
        seed_list_test.append(seed_test)

    print(len(seed_list_test))
    import time
    start_test = time.time()
    test_lesion = random_lesion_growth(seed_list_test, 400000)
    end_time = time.time()
    print(end_time - start_test)
    Functions.show_point_cloud_3d(test_lesion)
    visualize_simulated_clot(test_lesion)
    exit()
    run_func_get_lesion_seed()
