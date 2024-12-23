"""
The aim is to generate cube sequences for the detection of pulmonary embolism

see function "convert_ct_into_cubes"
Input: rescaled CT, blood vessel mask, airway mask, lung mask, mode
    mode is a string in ["blood_vessel", "airways", "lung", "all_ct"]
    in each mode, the sequence will be evenly sampled for these semantic
Output a list of dict, each dict with key:
'ct_data', the cube of rescaled CT, in shape [a, b, c] numpy float32;
'penalty_weight', the cube of penalty weight for mis-prediction of each voxel, in shape [a, b, c, 4] numpy float32;
channel 0 for blood vessels, channel 1 for airways, channel 2 for pulmonary parenchyma, channel 3 for others
'location_offset', the offset from the center of the blood vessel, tuple like (x, y, z)


"""
import visualization.visualize_3d.visualize_stl as stl
import Tool_Functions.Functions as Functions
import numpy as np
import os
import analysis.center_line_and_depth_3D as get_center_line
import format_convert.spatial_normalize as spatial_normalize


def convert_ct_into_tubes(rescaled_ct, vessel_mask, airway_mask, lung_mask, mode="blood_vessel",
                          absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5), max_cube_count=np.inf,
                          min_depth=3, show=True, return_check=True, shift=(0, 0, 0), penalty_array=None):
    """
    :param penalty_array:
    :param return_check: whether return a array same shape with rescaled_ct, indicate where we extract cubes.
    :param mode: is a string in ["blood_vessel", "airways", "lung", "all_ct"]
    :param lung_mask:
    :param airway_mask:
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
    assert mode in ["blood_vessel", "airways", "lung", "all_ct"]
    if show:
        print("mode:", mode)
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

    depth_mask = get_center_line.get_surface_distance(vessel_mask)
    assert min_depth >= 1
    location_array = np.where(depth_mask > (min_depth - 0.5))

    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))
    if show:
        print("mass center:", mass_center)

    if penalty_array is None:
        penalty_array = get_penalty_array(rescaled_ct, vessel_mask, airway_mask, lung_mask, show=show)

    if mode == 'all_ct':
        return pipeline_extract_all(rescaled_ct, penalty_array, cube_length, mass_center, shift,
                                    target_shape=target_shape, return_check=return_check, max_cube_count=max_cube_count)
    if mode == 'lung':
        return pipeline_extract_mask(rescaled_ct, penalty_array, cube_length, lung_mask, mass_center, shift,
                                     target_shape=target_shape, return_check=return_check,
                                     max_cube_count=max_cube_count)
    if mode == 'airways':
        return pipeline_extract_mask(rescaled_ct, penalty_array, cube_length, airway_mask, mass_center, shift,
                                     target_shape=target_shape, return_check=return_check,
                                     max_cube_count=max_cube_count)
    if mode == 'blood_vessel':
        return pipeline_extract_mask(rescaled_ct, penalty_array, cube_length, vessel_mask, mass_center, shift,
                                     target_shape=target_shape, return_check=return_check,
                                     max_cube_count=max_cube_count)


def convert_ct_into_tubes_inference(rescaled_ct, vessel_depth_mask, sample_region,
                                    absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5), max_cube_count=np.inf,
                                    min_depth=3, show=True, return_check=False, shift=(0, 0, 0), step=None):
    """
    :param return_check: whether return a array same shape with rescaled_ct, indicate where we extract cubes.
    :param sample_region: the region to sample the cubes, binary mask
    :param show:
    :param rescaled_ct: in shape [512, 512, 512], each voxel with resolution [334/512, 334/512, 1] mm^3
    :param vessel_depth_mask: numpy array, same shape with rescaled_ct, it is the return of
    get_center_line.get_surface_distance(vessel_mask)
    :param absolute_cube_length: the side length for the cube, in millimeters
    :param target_shape: the shape of the extracted cubes
    :param max_cube_count:
    :param min_depth: if the cube did not contain encoding_depth >= min_depth, it will be discarded, unless we cannot reach
    the num_cube_slice
    :param shift
    :param step, tuple like (x_step, y_step, z_step) if is None, step equals to the cube size
    :return: a list, each element is the return_dict of function "extract_cube"
    """

    assert np.shape(rescaled_ct) == np.shape(vessel_depth_mask) and len(np.shape(rescaled_ct)) == 3
    max_depth = np.max(vessel_depth_mask)
    if show:
        print("vessel with max encoding_depth:", max_depth)
    assert max_depth > min_depth + 1 and max_depth > 5

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

    assert min_depth >= 1
    location_array = np.where(vessel_depth_mask > (min_depth - 0.5))

    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))
    if show:
        print("mass center:", mass_center)

    return pipeline_extract_mask(rescaled_ct, None, cube_length, sample_region, mass_center, shift, step,
                                 target_shape=target_shape, return_check=return_check,
                                 max_cube_count=max_cube_count)


def pipeline_extract_all(rescaled_ct, penalty_array, cube_length, mass_center, shift=(0, 0, 0),
                         step=None, max_cube_count=np.inf, target_shape=(5, 5, 5), return_check=True):
    """
    like 3D convolution to extract cubes that inside lung
    :param mass_center:
    :param rescaled_ct:
    :param penalty_array:
    :param cube_length:
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

    list_sample = []

    x_min, x_max = cube_radius_x, shape_ct[0] - cube_radius_x - 1
    y_min, y_max = cube_radius_y, shape_ct[1] - cube_radius_y - 1
    z_min, z_max = cube_radius_z, shape_ct[2] - cube_radius_z - 1

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
                if num_sample > max_cube_count:
                    break
                center_location = (x, y, z)  # center_location for the cube
                center_location_offset = (center_location[0] - mass_center[0], center_location[1] - mass_center[1],
                                          center_location[2] - mass_center[2])
                sample = extract_cube(rescaled_ct, penalty_array, cube_length, center_location,
                                      center_location_offset, target_shape, depth_array=None)
                list_sample.append(sample)
                num_sample += 1

                if return_check:
                    x_start, x_end = center_location[0] - cube_radius_x, center_location[0] + cube_radius_x + 1
                    y_start, y_end = center_location[1] - cube_radius_y, center_location[1] + cube_radius_y + 1
                    z_start, z_end = center_location[2] - cube_radius_z, center_location[2] + cube_radius_z + 1
                    extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] = \
                        extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] + 1

    if return_check:
        return list_sample, extracted_count_mask
    return list_sample


def pipeline_extract_mask(rescaled_ct, penalty_array, cube_length, semantic_mask, mass_center, shift=(0, 0, 0),
                          step=None, max_cube_count=np.inf, target_shape=(5, 5, 5), return_check=True, show=True,
                          bounding_box=None):
    """
    like 3D convolution to extract cubes that inside lung
    :param mass_center:
    :param rescaled_ct:
    :param penalty_array:
    :param cube_length:
    :param semantic_mask:
    :param shift: shift when making the 3D grid
    :param step: by default it will let step=cube_length
    :param max_cube_count: the max cube extracted
    :param target_shape: the resized shape for the extracted cube
    :param return_check: whether return the extracted_count_mask
    :param show
    :param bounding_box: the bounding_box of the semantic
    :return: a list of dict, each dict is the return of "extract_cube"
    """
    if return_check:
        extracted_count_mask = np.zeros(np.shape(rescaled_ct), 'float32')
    else:
        extracted_count_mask = None

    shape_ct = np.shape(rescaled_ct)
    if show:
        print("the ct with shape:", shape_ct)
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    if bounding_box is None:
        bounding_box = Functions.get_bounding_box(semantic_mask, pad=int(min(cube_length) / 2))

    if show:
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
                if semantic_mask[x, y, z] < 0.5:  # the center_location outside the semantic mask
                    continue
                if num_sample > max_cube_count:
                    break
                center_location = (x, y, z)  # center_location for the cube
                center_location_offset = (center_location[0] - mass_center[0], center_location[1] - mass_center[1],
                                          center_location[2] - mass_center[2])
                sample = extract_cube(rescaled_ct, penalty_array, cube_length, center_location,
                                      center_location_offset, target_shape, depth_array=None)
                list_sample.append(sample)
                num_sample += 1

                if return_check:
                    x_start, x_end = center_location[0] - cube_radius_x, center_location[0] + cube_radius_x + 1
                    y_start, y_end = center_location[1] - cube_radius_y, center_location[1] + cube_radius_y + 1
                    z_start, z_end = center_location[2] - cube_radius_z, center_location[2] + cube_radius_z + 1
                    extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] = \
                        extracted_count_mask[x_start: x_end, y_start: y_end, z_start: z_end] + 1
    if return_check:
        return list_sample, extracted_count_mask
    return list_sample


def extract_cube(rescaled_ct, penalty_array, cube_length, center_location, center_location_offset,
                 target_shape=None, depth_array=None):
    """
    :param depth_array:
    :param rescaled_ct: numpy float32 array with shape like (512, 512, 512)
    :param penalty_array: in shape (cube_length[0], cube_length[1], cube_length[2], 4)
    :param cube_length: an tuple of int, mod 2 == 1, like (11, 11, 7)
    :param center_location: the absolute location of the cube center, like (256, 325, 178)
    :param center_location_offset: the offset of the cube center to the vessel mass center, like (13, 17, 55)
    :param target_shape: rescale te cube to the target_shape
    :return: dict with key 'ct_data', 'penalty_weight', 'given_vector', 'location_offset'
    """
    assert cube_length[0] % 2 == 1 and cube_length[1] % 2 == 1 and cube_length[2] % 2 == 1 and min(cube_length) > 0
    assert np.shape(penalty_array) == (np.shape(rescaled_ct)[0], np.shape(rescaled_ct)[1],
                                       np.shape(rescaled_ct)[2], 4) or penalty_array is None
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    ct_cube = rescaled_ct[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                          center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                          center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1]

    if penalty_array is not None:
        penalty_cube = penalty_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                                     center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                                     center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1, :]
    else:
        penalty_cube = None

    if depth_array is not None:  # here add one for depth_cube * (penalty_cube[:, :, :, 0] * weight_0 + ...)
        depth_cube = depth_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                                 center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                                 center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] + 1
    else:
        depth_cube = np.ones(np.shape(ct_cube), 'float32')

    if target_shape is not None:
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, target_shape)
        depth_cube = spatial_normalize.rescale_to_new_shape(depth_cube, target_shape)

        if penalty_array is not None:
            new_penalty_array = np.zeros((target_shape[0], target_shape[1], target_shape[2], 4), 'float32')
            for z_index in range(4):
                new_penalty_array[:, :, :, z_index] = \
                    spatial_normalize.rescale_to_new_shape(penalty_cube[:, :, :, z_index], target_shape)
            penalty_cube = new_penalty_array

    return_dict = {'ct_data': ct_cube, 'penalty_weight': penalty_cube, 'location_offset': center_location_offset,
                   'given_vector': None, 'center_location': center_location, 'depth_cube': depth_cube}

    return return_dict


def get_quantile(data_cube, quantile_list, remove_numbers=None):
    """

    :param data_cube: an numpy array
    :param quantile_list: like [0, 10, 20, ..., 100]
    :param remove_numbers: if true, remove corresponding numbers
    :return: the list for quantile
    """

    assert max(quantile_list) <= 100 and min(quantile_list) >= 0

    data_cube = np.reshape(data_cube, [-1, ])
    num_voxels = len(data_cube)
    remove_count = 0
    if remove_numbers is not None:
        for i in range(num_voxels):
            if data_cube[i] in remove_numbers:
                data_cube[i] = np.inf
                remove_count += 1
    data_cube.sort()

    remained_voxels = len(data_cube) - remove_count

    return_list = []

    for quantile in quantile_list:
        if quantile < 100:
            return_list.append(data_cube[int(remained_voxels * quantile / 100)])
        else:
            return_list.append(data_cube[-1])
    return return_list


def get_penalty_array(rescaled_ct, vessel_mask, airway_mask, lung_mask, show=False):
    """

    :param rescaled_ct:
    :param vessel_mask:
    :param airway_mask:
    :param lung_mask:
    :param show:
    :return: penalty array in numpy float32 [x, y, z, 4]
    channel 0 for blood vessels, channel 1 for airways, channel 2 for pulmonary parenchyma, channel 3 for others
    the mask can be extracted from penalty array, like airway_mask = np.array(penalty_array[:, :, :, 1] > 0)
    """
    std_overall = np.std(rescaled_ct)
    if show:
        print("overall std:", std_overall)
    std_vessel = get_std_for_region(rescaled_ct, vessel_mask, show=show)
    std_airway = get_std_for_region(rescaled_ct, airway_mask, show=show)
    std_lung_mask = get_std_for_region(rescaled_ct, lung_mask, show=show)

    shape_ct = np.shape(rescaled_ct)
    total_voxel = shape_ct[0] * shape_ct[1] * shape_ct[2]
    penalty_array = np.zeros((shape_ct[0], shape_ct[1], shape_ct[2], 4), 'float32')

    total_penalty = 1000000
    base_penalty = total_penalty / total_voxel

    # penalty sum for each semantic is: total_penalty / std_semantic * std_overall
    penalty_array[:, :, :, 3] = base_penalty
    penalty_array[:, :, :, 2] = \
        base_penalty * std_overall / std_lung_mask * total_voxel / np.sum(lung_mask) * lung_mask
    penalty_array[:, :, :, 1] = \
        base_penalty * std_overall / std_airway * total_voxel / np.sum(airway_mask) * airway_mask
    penalty_array[:, :, :, 0] = \
        base_penalty * std_overall / std_vessel * total_voxel / np.sum(vessel_mask) * vessel_mask

    return penalty_array


def get_std_for_region(rescaled_ct, region_mask, trim=0.2, show=False):
    """

    :param show:
    :param rescaled_ct:
    :param region_mask: must be binary
    :param trim:
    :return:
    """
    assert trim < 0.5
    total_voxel = np.sum(region_mask)
    max_value = np.max(rescaled_ct)
    rescaled_ct = rescaled_ct + (1 - region_mask) * max_value
    rescaled_ct = np.reshape(rescaled_ct, [-1, ])
    rescaled_ct = np.sort(rescaled_ct)

    std_value = np.std(rescaled_ct[int(total_voxel * trim): int(total_voxel * (1 - trim))])
    if show:
        print("standard deviation for the region is:", std_value)
    return std_value


def pipeline_process():

    top_dict_dataset = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/list_pickle_dataset/'
    top_dict_extract_count_mask = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/extract_mask_for_check/'

    top_dict_normal = '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/normal_scan_extended/'
    top_dict_semantic = '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/'

    list_file_name = os.listdir(top_dict_normal)

    # list_file_name.remove('Scanner-B_B21.npy')

    processed_count = 0
    for file_name in list_file_name:
        print("\nprocessing:", file_name, len(list_file_name) - processed_count, 'left')

        if os.path.exists(top_dict_extract_count_mask + 'all_ct/' + file_name[:-4] + '.npz'):
            print('processed')
            processed_count += 1
            continue

        rescaled_ct = np.load(top_dict_normal + file_name)
        lung_mask = np.load(top_dict_semantic + 'lung_mask/' + file_name[:-4] + '.npz')['array']
        artery_mask = np.load(top_dict_semantic + 'artery_mask/' + file_name[:-4] + '.npz')['array']
        vein_mask = np.load(top_dict_semantic + 'vein_mask/' + file_name[:-4] + '.npz')['array']
        blood_robust_mask = np.load(top_dict_semantic + 'blood_mask/' + file_name[:-4] + '.npz')['array']

        vessel_mask = np.array(artery_mask + vein_mask + blood_robust_mask > 0.5, 'float32')

        airway_mask = np.load(top_dict_semantic + 'airway_mask/' + file_name[:-4] + '.npz')['array']

        stl.save_numpy_as_stl(vessel_mask,
                              '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check'
                              '/vessels/', file_name[:-4] + '.stl')
        stl.save_numpy_as_stl(airway_mask,
                              '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check'
                              '/airways/', file_name[:-4] + '.stl')

        penalty_array = get_penalty_array(rescaled_ct, vessel_mask, airway_mask, lung_mask, show=True)

        list_sample_blood, extract_count_mask_blood = convert_ct_into_tubes(rescaled_ct, vessel_mask, airway_mask,
                                                                            lung_mask, mode="blood_vessel",
                                                                            penalty_array=penalty_array)
        print("num_blood_vessel_cubes:", len(list_sample_blood))
        save_dict_pickle_vessel = top_dict_dataset + 'blood_vessels/'
        save_dict_check_array_vessel = top_dict_extract_count_mask + 'blood_vessels/'
        Functions.pickle_save_object(save_dict_pickle_vessel + file_name[:-4] + '.pickle', list_sample_blood)
        Functions.save_np_array(save_dict_check_array_vessel, file_name[:-4] + '.npz', extract_count_mask_blood, True)

        list_sample_airways, extract_count_mask_airways = convert_ct_into_tubes(rescaled_ct, vessel_mask, airway_mask,
                                                                                lung_mask, mode="airways",
                                                                                penalty_array=penalty_array)
        print("num_airway_cubes:", len(list_sample_airways))
        save_dict_pickle_airway = top_dict_dataset + 'airways/'
        save_dict_check_array_airway = top_dict_extract_count_mask + 'airways/'
        Functions.pickle_save_object(save_dict_pickle_airway + file_name[:-4] + '.pickle', list_sample_airways)
        Functions.save_np_array(save_dict_check_array_airway, file_name[:-4] + '.npz', extract_count_mask_airways, True)

        list_sample_lung, extract_count_mask_lung = convert_ct_into_tubes(rescaled_ct, vessel_mask, airway_mask,
                                                                          lung_mask, mode="lung",
                                                                          penalty_array=penalty_array)
        print("num_lung_cubes:", len(list_sample_lung))
        save_dict_pickle_lung = top_dict_dataset + 'lung_region/'
        save_dict_check_array_lung = top_dict_extract_count_mask + 'lung_region/'
        Functions.pickle_save_object(save_dict_pickle_lung + file_name[:-4] + '.pickle', list_sample_lung)
        Functions.save_np_array(save_dict_check_array_lung, file_name[:-4] + '.npz', extract_count_mask_lung, True)

        list_sample_all, extract_count_mask_all = convert_ct_into_tubes(rescaled_ct, vessel_mask, airway_mask,
                                                                        lung_mask, mode="all_ct",
                                                                        penalty_array=penalty_array)
        print("num_all_cubes:", len(list_sample_all))
        save_dict_pickle_all = top_dict_dataset + 'all_ct/'
        save_dict_check_array_all = top_dict_extract_count_mask + 'all_ct/'
        Functions.pickle_save_object(save_dict_pickle_all + file_name[:-4] + '.pickle', list_sample_all)
        Functions.save_np_array(save_dict_check_array_all, file_name[:-4] + '.npz', extract_count_mask_all, True)

        processed_count += 1


if __name__ == '__main__':
    pipeline_process()

    exit()
