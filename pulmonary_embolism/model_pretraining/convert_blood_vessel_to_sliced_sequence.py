"""
this version only slice blood vessel, but each sample contains more information, like encoding_depth mask, the high resolution
cube in [11, 11, 11]

if the cube did not contain encoding_depth >= min_depth, it will be discarded

pipeline_extract_mask_v1: get cube like 3D convolution
pipeline_extract_mask_v2: get cube on center line

"""
import Tool_Functions.Functions as Functions
import numpy as np
import random
import os
import analysis.center_line_and_depth_3D as center_line_and_depth
import format_convert.spatial_normalize as spatial_normalize


def convert_ct_into_tubes(rescaled_ct, vessel_mask, absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5),
                          max_cube_count=np.inf, min_depth=4, show=True, return_check=False, shift=(0, 0, 0),
                          step=None, only_v1=False, only_v2=False, depth_array=None, mass_center=None):
    """
    v1, get sample like convolution; v2, get sample on center line

    :param mass_center:
    :param depth_array:
    :param only_v2:
    :param only_v1: do not return center line
    :param step:
    :param return_check: whether return a array same shape with rescaled_ct, indicate where we extract cubes.
    :param show:
    :param rescaled_ct: in shape [512, 512, 512], each voxel with resolution [334/512, 334/512, 1] mm^3
    :param vessel_mask: binary numpy array, same shape with rescaled_ct
    :param absolute_cube_length: the side length for the cube, in millimeters
    :param target_shape: the shape of the extracted cubes
    :param max_cube_count:
    :param min_depth: if the cube did not contain encoding_depth >= min_depth, it will be discarded, unless we cannot reach
    the num_cube_slice
    :param shift
    :return: a list, each element is the return_dict of function "extract_cube"
    """

    assert np.shape(rescaled_ct) == np.shape(vessel_mask) and len(np.shape(rescaled_ct)) == 3

    cube_length = []

    if round(absolute_cube_length[0] / 334 * 512) % 2 == 0:
        cube_length.append(round(absolute_cube_length[0] / 334 * 512) + 1)
    else:
        cube_length.append(round(absolute_cube_length[0] / 334 * 512))

    if round(absolute_cube_length[1] / 334 * 512) % 2 == 0:
        cube_length.append(round(absolute_cube_length[1] / 334 * 512) + 1)
    else:
        cube_length.append(round(absolute_cube_length[1] / 334 * 512))

    if round(absolute_cube_length[2]) % 2 == 0:
        cube_length.append(round(absolute_cube_length[2]) + 1)
    else:
        cube_length.append(round(absolute_cube_length[2]))

    if show:
        print("cube_length:", cube_length)

    if depth_array is None:
        depth_array = center_line_and_depth.get_surface_distance(vessel_mask)

    if mass_center is None:
        location_array = np.where(depth_array > (min_depth - 0.5))

        mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                       int(np.average(location_array[2])))
        if show:
            print("mass center:", mass_center)

    if show:
        max_depth = np.max(depth_array)
        print("max encoding_depth:", max_depth)

    if only_v1 and return_check:
        sample_sequence_v1, mask_check_v1 = \
            extract_vessel_sequence_v1(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                       max_cube_count, target_shape, return_check)
        return sample_sequence_v1, mask_check_v1
    if only_v1 and not return_check:
        sample_sequence_v1 = extract_vessel_sequence_v1(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                                        max_cube_count, target_shape, return_check)
        return sample_sequence_v1

    if only_v2 and return_check:
        center_line_mask = center_line_and_depth.get_center_line(vessel_mask,
                                                                 surface_distance=depth_array) * vessel_mask
        sample_sequence_v2, mask_check_v2 = \
            extract_vessel_sequence_v2(rescaled_ct, cube_length, depth_array, mass_center,
                                       center_line_mask, min_depth=min_depth, return_check=return_check)
        return sample_sequence_v2, mask_check_v2
    if only_v2 and not return_check:
        center_line_mask = center_line_and_depth.get_center_line(vessel_mask,
                                                                 surface_distance=depth_array) * vessel_mask
        sample_sequence_v2 = extract_vessel_sequence_v2(rescaled_ct, cube_length, depth_array, mass_center,
                                                        center_line_mask, min_depth=min_depth)
        return sample_sequence_v2

    if not return_check:
        center_line_mask = center_line_and_depth.get_center_line(vessel_mask,
                                                                 surface_distance=depth_array) * vessel_mask

        sample_sequence_v1 = extract_vessel_sequence_v1(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                                        max_cube_count, target_shape, return_check)
        sample_sequence_v2 = extract_vessel_sequence_v2(rescaled_ct, cube_length, depth_array, mass_center,
                                                        center_line_mask, min_depth=min_depth)
        return sample_sequence_v1, sample_sequence_v2

    center_line_mask = center_line_and_depth.get_center_line(vessel_mask, surface_distance=depth_array) * vessel_mask

    sample_sequence_v1, mask_check_v1 = \
        extract_vessel_sequence_v1(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                   max_cube_count, target_shape, return_check)
    sample_sequence_v2, mask_check_v2 = \
        extract_vessel_sequence_v2(rescaled_ct, cube_length, depth_array, mass_center,
                                   center_line_mask, min_depth=min_depth, return_check=return_check)

    return sample_sequence_v1, mask_check_v1, sample_sequence_v2, mask_check_v2


def extract_vessel_sequence_v1(rescaled_ct, cube_length, depth_array, mass_center, shift=(0, 0, 0),
                               step=None, max_cube_count=np.inf, target_shape=(5, 5, 5), return_check=False):
    """
    like 3D convolution to extract cubes that inside lung
    :param mass_center:
    :param rescaled_ct:
    :param cube_length: (x_length, y_length, z_length)
    :param depth_array:
    :param shift: shift when making the 3D grid
    :param step: by default it will let step=cube_length
    :param max_cube_count: the max cube extracted
    :param target_shape: the resized shape for the extracted cube
    :param return_check: whether return the extracted_count_mask
    :return: a list of dict, each dict is the return of "extract_cube"
    """
    if return_check:
        extracted_count_mask = np.zeros(np.shape(rescaled_ct), 'float32')
    else:
        extracted_count_mask = None

    shape_ct = np.shape(rescaled_ct)
    print("the ct with shape:", shape_ct)
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    bounding_box = Functions.get_bounding_box(depth_array, pad=int(min(cube_length) / 2))
    print("the bounding box for this semantic is:", bounding_box)

    list_sample = []

    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]

    x_min += shift[0]
    x_max += shift[0]
    y_min += shift[1]
    y_max += shift[1]
    z_min += shift[2]
    z_max += shift[2]

    x_min, x_max = max(cube_radius_x, x_min), min(shape_ct[0] - cube_radius_x - 1, x_max)
    y_min, y_max = max(cube_radius_y, y_min), min(shape_ct[1] - cube_radius_y - 1, y_max)
    z_min, z_max = max(cube_radius_z, z_min), min(shape_ct[2] - cube_radius_z - 1, z_max)

    print("x_start", x_min, " x_end", x_max, "y_start", y_min, "y_end", y_max, "z_start", z_min, "z_end", z_max)

    if step is None:
        step = cube_length
    assert min(step) >= 1

    num_sample = 0
    for x in range(x_min, x_max, step[0]):
        for y in range(y_min, y_max, step[1]):
            for z in range(z_min, z_max, step[2]):
                if depth_array[x, y, z] < 0.5:  # the central_location outside the vessel mask
                    continue
                if num_sample > max_cube_count:
                    break
                central_location = (x, y, z)  # central_location for the cube
                central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                           central_location[2] - mass_center[2])

                sample = extract_cube(rescaled_ct, depth_array, cube_length, central_location,
                                      central_location_offset, target_shape)

                list_sample.append(sample)
                num_sample += 1

                if return_check:
                    x_start, x_end = central_location[0] - cube_radius_x, central_location[0] + cube_radius_x + 1
                    y_start, y_end = central_location[1] - cube_radius_y, central_location[1] + cube_radius_y + 1
                    z_start, z_end = central_location[2] - cube_radius_z, central_location[2] + cube_radius_z + 1
                    extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] = \
                        extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] + 1
    if return_check:
        return list_sample, extracted_count_mask
    return list_sample


def extract_vessel_sequence_v2(rescaled_ct, cube_length, depth_array, mass_center, center_line_mask,
                               cube_count=800, min_depth=4, target_shape=(5, 5, 5), return_check=False):
    """

    :param min_depth:
    :param cube_count:
    :param center_line_mask:
    :param mass_center:
    :param rescaled_ct:
    :param cube_length: (x_length, y_length, z_length)
    :param depth_array:
    :param target_shape: the resized shape for the extracted cube
    :param return_check: whether return the extracted_count_mask
    :return: a list of dict, each dict is the return of "extract_cube"
    """
    if return_check:
        extracted_count_mask = np.zeros(np.shape(rescaled_ct), 'float32')
    else:
        extracted_count_mask = None

    shape_ct = np.shape(rescaled_ct)
    print("the ct with shape:", shape_ct)
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    list_sample = []

    center_line_locations = Functions.get_location_list(np.where(center_line_mask > 0.5))

    print("the center line has:", len(center_line_locations), 'voxels')

    qualified_location_list = []

    for loc in center_line_locations:
        if depth_array[loc] >= min_depth:
            qualified_location_list.append(loc)

    print("there are", len(qualified_location_list), "center line voxel with encoding_depth >=", min_depth)

    qualified_location_mask = np.zeros(shape_ct, 'float32')
    qualified_location_mask[Functions.get_location_array(qualified_location_list)] = 1

    random.shuffle(qualified_location_list)

    for index in range(0, min(len(qualified_location_list), cube_count)):

        central_location = qualified_location_list[index]  # central_location for the cube
        central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                   central_location[2] - mass_center[2])

        sample = extract_cube(rescaled_ct, depth_array, cube_length, central_location,
                              central_location_offset, target_shape)

        list_sample.append(sample)

        x_start, x_end = central_location[0] - cube_radius_x, central_location[0] + cube_radius_x + 1
        y_start, y_end = central_location[1] - cube_radius_y, central_location[1] + cube_radius_y + 1
        z_start, z_end = central_location[2] - cube_radius_z, central_location[2] + cube_radius_z + 1

        if return_check:

            extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] = \
                extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] + 1

        qualified_location_mask[x_start: x_end, y_start: y_end, z_start: z_end] = 0  # this means location is included

    remaining_qualified_loc_set = set(Functions.get_location_list(np.where(qualified_location_mask == 1)))
    # avoid missing qualified locations

    while len(remaining_qualified_loc_set) > 0:
        central_location = remaining_qualified_loc_set.pop()
        remove_list = []
        for loc in remaining_qualified_loc_set:
            if abs(loc[0] - central_location[0]) < cube_radius_x:
                remove_list.append(loc)
            elif abs(loc[1] - central_location[1]) < cube_radius_y:
                remove_list.append(loc)
            elif abs(loc[2] - central_location[2]) < cube_radius_z:
                remove_list.append(loc)
        for loc in remove_list:
            remaining_qualified_loc_set.remove(loc)

        print('add', central_location)
        central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                   central_location[2] - mass_center[2])

        sample = extract_cube(rescaled_ct, depth_array, cube_length, central_location,
                              central_location_offset, target_shape)

        list_sample.append(sample)

        x_start, x_end = central_location[0] - cube_radius_x, central_location[0] + cube_radius_x + 1
        y_start, y_end = central_location[1] - cube_radius_y, central_location[1] + cube_radius_y + 1
        z_start, z_end = central_location[2] - cube_radius_z, central_location[2] + cube_radius_z + 1

        if return_check:
            extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] = \
                extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] + 1

    print("extracted:", len(list_sample), 'samples')

    if return_check:
        return list_sample, extracted_count_mask
    return list_sample


def extract_cube(rescaled_ct, depth_array, cube_length, central_location, central_location_offset,
                 target_shape=None):
    """
    :param depth_array:
    :param rescaled_ct: numpy float32 array with shape like (512, 512, 512)
    :param cube_length: an tuple of int, mod 2 == 1, like (11, 11, 7)
    :param central_location: the absolute location of the cube center, like (256, 325, 178)
    :param central_location_offset: the offset of the cube center to the vessel mass center, like (13, 17, 55)
    :param target_shape: rescale te cube to the target_shape
    :return: dict with key 'ct_data', 'penalty_weight', 'given_vector', 'location_offset'
    """
    assert cube_length[0] % 2 == 1 and cube_length[1] % 2 == 1 and cube_length[2] % 2 == 1 and min(cube_length) > 0

    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    ct_cube = rescaled_ct[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                          central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                          central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    depth_cube = depth_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                             central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                             central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    if target_shape is not None:
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, target_shape)
        depth_cube = spatial_normalize.rescale_to_new_shape(depth_cube, target_shape)

    return_dict = {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
                   'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube}

    return return_dict


def pipeline_process(de_noise=False, only_v1=True, high_resolution=False, load_func_ct=None, fold=(0, 1)):
    import chest_ct_database.feature_manager.save_as_float_16 as convert_to_float16
    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
    else:
        absolute_cube_length = (4, 4, 5)

    import collaborators_package.denoise_chest_ct.denoise_predict as de_noising
    if de_noise:
        de_noise_model = de_noising.load_model()
    else:
        de_noise_model = None

    top_dict_dataset = '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution_denoise_new/RAD_3615/'

    top_dict_ct = '/data_disk/RAD-ChestCT_dataset/stack_ct_rad_format/'
    top_dict_semantic = '/data_disk/RAD-ChestCT_dataset/semantic_in_rescaled_ct/'

    de_noise_save_dict = '/data_disk/RAD-ChestCT_dataset/rescaled_ct_denoise_float16/'

    list_file_name = os.listdir(top_dict_ct)[fold[0]::fold[1]]

    save_dict_pickle_v1 = top_dict_dataset + 'v1/'
    save_dict_pickle_v2 = top_dict_dataset + 'v2/'

    processed_count = 0
    for file_name in list_file_name:
        if file_name in ['xwzc000104_unknown.npy']:
            print("wrong scan")
            processed_count += 1
            continue
        print("\nprocessing:", file_name, len(list_file_name) - processed_count, 'left')

        if only_v1:
            if os.path.exists(top_dict_dataset + 'v1/' + file_name[:-4] + '.pickle'):
                print('processed')
                processed_count += 1
                continue
        elif os.path.exists(top_dict_dataset + 'v2/' + file_name[:-4] + '.pickle'):
            print('processed')
            processed_count += 1
            continue

        if de_noise:
            assert de_noise_save_dict is not None
            if os.path.exists(os.path.join(de_noise_save_dict, file_name[:-4] + '.npz')):
                print("loading denoise ct")
                rescaled_ct = np.load(os.path.join(de_noise_save_dict, file_name[:-4] + '.npz'))['array']
            else:
                if load_func_ct is None:
                    rescaled_ct = np.load(top_dict_ct + file_name)
                else:
                    rescaled_ct = load_func_ct(top_dict_ct + file_name)
                rescaled_ct = de_noising.denoise_rescaled_array(rescaled_ct, de_noise_model)
                rescaled_ct = convert_to_float16.convert_rescaled_ct_to_float16(rescaled_ct)
                print("saving denoise rescaled array to", os.path.join(de_noise_save_dict, file_name[:-4] + '.npz'))
                Functions.save_np_array(de_noise_save_dict, file_name[:-4], rescaled_ct, compress=True)
        else:
            if load_func_ct is None:
                rescaled_ct = np.load(top_dict_ct + file_name)
            else:
                rescaled_ct = load_func_ct(top_dict_ct + file_name)

        vessel_mask = np.load(top_dict_semantic + 'blood_mask/' + file_name[:-4] + '.npz')['array']
        airway_mask = np.load(top_dict_semantic + 'airway_mask/' + file_name[:-4] + '.npz')['array']

        if np.sum(airway_mask) < 1000:
            print("wrong seg")
            processed_count += 1
            continue

        if only_v1:
            sample_list_v1 = convert_ct_into_tubes(
                rescaled_ct, vessel_mask, absolute_cube_length=absolute_cube_length, only_v1=only_v1)
            Functions.pickle_save_object(save_dict_pickle_v1 + file_name[:-4] + '.pickle', sample_list_v1)
            processed_count += 1
            continue

        sample_list_v1, sample_list_v2 = convert_ct_into_tubes(rescaled_ct, vessel_mask,
                                                               absolute_cube_length=absolute_cube_length)

        Functions.pickle_save_object(save_dict_pickle_v1 + file_name[:-4] + '.pickle', sample_list_v1)
        Functions.pickle_save_object(save_dict_pickle_v2 + file_name[:-4] + '.pickle', sample_list_v2)

        processed_count += 1


if __name__ == '__main__':
    from chest_ct_database.public_datasets.RAD_ChestCT_dataset import load_func_for_ct

    pipeline_process(de_noise=True, only_v1=True, high_resolution=True, load_func_ct=load_func_for_ct, fold=(0, 3))
