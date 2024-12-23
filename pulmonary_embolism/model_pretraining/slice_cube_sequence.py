"""
The aim is to generate cube sequences for the detection of pulmonary embolism

Input: blood vessel mask (artery or vein, or whole blood vessel), and the rescaled CT
Output a list of dict, each dict with key:
'ct_data', the cube of rescaled CT, in shape [a, b] numpy float32;
'penalty_weight', the cube of penalty weight for mis-prediction of each voxel, in shape [a, b] numpy float32;
'given_vector', number of vessel voxels, center encoding_depth, quantile 0, 10, 20, ..., 100 of encoding_depth;
center ct signal, quantile 0, 10, 20, ..., 100 of ct signal, ct signal std;

Note the positional embedding will be added to the feature vector, so each cube will be:
concatenate(given_vector, feature_vector + positional_embedding)
"feature_vector" is abstracted by convolution layers with trainable parameters
"""
import Tool_Functions.Functions as Functions
import numpy as np
import analysis.center_line_and_depth_3D as get_center_line
import format_convert.spatial_normalize as spatial_normalize


def extract_cube_sequence(rescaled_ct, vessel_mask, absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5),
                          max_num_cube_slice=2048, min_depth=3, show=False):
    """

    Select criteria:
    Cube center will never inside another cube
    First to select cube with center voxel of the largest encoding_depth

    :param show:
    :param rescaled_ct: in shape [512, 512, 512], each voxel with resolution [334/512, 334/512, 1] mm^3
    :param vessel_mask: binary numpy array, same shape with rescaled_ct
    :param absolute_cube_length: the side length for the cube, in millimeters
    :param target_shape: the shape of the extracted cubes
    :param max_num_cube_slice: how many cubes for the vessel
    :param min_depth: if the cube did not contain encoding_depth >= min_depth, it will be discarded, unless we cannot reach
    the num_cube_slice
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
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    depth_mask = get_center_line.get_surface_distance(vessel_mask)

    location_array = np.where(depth_mask > (min_depth - 0.5))

    location_set = set(Functions.get_location_list(location_array))

    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))
    if show:
        print("mass center:", mass_center)

    return_list = []

    temp_array = np.array(vessel_mask > 0.5, 'float32')

    cube_count = 0
    while True:
        if cube_count >= max_num_cube_slice:
            break
        if len(location_set) == 0:
            break

        max_depth = 0
        central_location = None
        for loc in location_set:
            if depth_mask[loc] > max_depth:
                central_location = loc
                max_depth = depth_mask[loc]

        if max_depth < min_depth:
            break

        central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                   central_location[2] - mass_center[2])

        if cube_count % 20 == 0 and show:
            print("cube_count_base:", cube_count)
            print("current_depth:", max_depth)
            print("current_location:", central_location)
            print("location_offset:", central_location_offset)

        current_cube_dict = extract_cube(rescaled_ct, vessel_mask, depth_mask, cube_length, central_location,
                                         central_location_offset, target_shape)
        return_list.append(current_cube_dict)

        vessel_mask_cube = vessel_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                       central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                       central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

        remaining_mask_cube = temp_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                         central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                         central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

        list_voxel_remove = Functions.get_location_list(np.where((vessel_mask_cube * remaining_mask_cube) > 0.5))

        for i in range(len(list_voxel_remove)):
            list_voxel_remove[i] = (list_voxel_remove[i][0] + central_location[0] - cube_radius_x,
                                    list_voxel_remove[i][1] + central_location[1] - cube_radius_y,
                                    list_voxel_remove[i][2] + central_location[2] - cube_radius_z)

        set_voxel_remove = set(list_voxel_remove)

        location_set.difference_update(set_voxel_remove)

        temp_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                   central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                   central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1] = 0
        cube_count += 1

    return return_list


def extract_cube_sequence_with_check(rescaled_ct, vessel_mask, absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5),
                                     max_num_cube_slice=2048, min_depth=3, show=False):
    """

    Select criteria:
    Cube center will never inside another cube
    First to select cube with center voxel of the largest encoding_depth

    :param show:
    :param rescaled_ct: in shape [512, 512, 512], each voxel with resolution [334/512, 334/512, 1] mm^3
    :param vessel_mask: binary numpy array, same shape with rescaled_ct
    :param absolute_cube_length: the side length for the cube, in millimeters
    :param target_shape: the shape of the extracted cubes
    :param max_num_cube_slice: how many cubes for the vessel
    :param min_depth: if the cube did not contain encoding_depth >= min_depth, it will be discarded, unless we cannot reach
    the num_cube_slice
    :return: a list, each element is the return_dict of function "extract_cube",
    a extracted_mask in shape [512, 512, 512], indicate the number of times a voxel is sliced into the cube_sequence.
    """
    assert np.shape(rescaled_ct) == np.shape(vessel_mask) and len(np.shape(rescaled_ct)) == 3

    extracted_mask = np.zeros([512, 512, 512], 'float32')

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
    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    depth_mask = get_center_line.get_surface_distance(vessel_mask)

    location_array = np.where(depth_mask > (min_depth - 0.5))

    location_set = set(Functions.get_location_list(location_array))

    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))
    if show:
        print("mass center:", mass_center)

    return_list = []

    temp_array = np.array(vessel_mask > 0.5, 'float32')

    cube_count = 0
    while True:
        if cube_count >= max_num_cube_slice:
            break
        if len(location_set) == 0:
            break

        max_depth = 0
        central_location = None
        for loc in location_set:
            if depth_mask[loc] > max_depth:
                central_location = loc
                max_depth = depth_mask[loc]

        if max_depth < min_depth:
            break

        central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                   central_location[2] - mass_center[2])

        if cube_count % 20 == 0 and show:
            print("cube_count_base:", cube_count)
            print("current_depth:", max_depth)
            print("current_location:", central_location)
            print("location_offset:", central_location_offset)

        current_cube_dict = extract_cube(rescaled_ct, vessel_mask, depth_mask, cube_length, central_location,
                                         central_location_offset, target_shape)
        return_list.append(current_cube_dict)

        vessel_mask_cube = vessel_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                       central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                       central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

        remaining_mask_cube = temp_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                         central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                         central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

        list_voxel_remove = Functions.get_location_list(np.where((vessel_mask_cube * remaining_mask_cube) > 0.5))

        for i in range(len(list_voxel_remove)):
            list_voxel_remove[i] = (list_voxel_remove[i][0] + central_location[0] - cube_radius_x,
                                    list_voxel_remove[i][1] + central_location[1] - cube_radius_y,
                                    list_voxel_remove[i][2] + central_location[2] - cube_radius_z)

        set_voxel_remove = set(list_voxel_remove)

        location_set.difference_update(set_voxel_remove)

        temp_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                   central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                   central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1] = 0

        extracted_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                       central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                       central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1] = \
            extracted_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                           central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                           central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1] + 1
        cube_count += 1

    return return_list, extracted_mask


def extract_cube(rescaled_ct, vessel_mask, depth_mask, cube_length, central_location, central_location_offset,
                 target_shape=None):
    """

    :param rescaled_ct:
    :param vessel_mask:
    :param depth_mask:
    :param cube_length: an tuple of int, mod 2 == 1, like (11, 11, 7)
    :param central_location: the absolute location of the cube center, like (256, 325, 178)
    :param central_location_offset: the offset of the cube center to the vessel center, like (13, 17, 55)
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

    vessel_mask_cube = vessel_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                   central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                   central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    depth_cube = depth_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                            central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                            central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    given_vector = [np.sum(vessel_mask_cube), depth_mask[central_location]]

    quantile_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    given_vector = given_vector + get_quantile(depth_cube, quantile_list)
    given_vector.append(rescaled_ct[central_location])
    given_vector = given_vector + get_quantile(ct_cube, quantile_list)
    given_vector.append(np.std(ct_cube))

    if target_shape is not None:
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, target_shape, change_format=True)
        vessel_mask_cube = spatial_normalize.rescale_to_new_shape(vessel_mask_cube, target_shape, change_format=True)

    return_dict = {'ct_data': ct_cube, 'penalty_weight': vessel_mask_cube, 'location_offset': central_location_offset,
                   'given_vector': given_vector}

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


if __name__ == '__main__':

    mask_extracted = np.load(
        '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/extracted_masks/depth_3/Scanner-A_A5.npz')['array']
    rescaled_ct = np.load('/home/zhoul0a/Desktop/normal_people/rescaled_ct_array/Scanner-A_A5.npy')
    for index in range(150, 350, 10):
        Functions.merge_image_with_mask(
            np.clip(rescaled_ct[:, :, index] + 0.5, 0, 1), mask_extracted[:, :, index])
    exit()
    rescaled_ct_normal = np.load('/home/zhoul0a/Desktop/absolutely_normal/rescaled_ct/Scanner-A_A1.npy')

    artery_mask = Functions.read_in_mha(
        '/home/zhoul0a/Desktop/absolutely_normal/rescaled_CT_central_axis/Scanner-A_A1_predict_artery.mha')

    cube_dict_list, mask_extracted = extract_cube_sequence_with_check(
        rescaled_ct_normal, artery_mask, max_num_cube_slice=2024, show=True, min_depth=4)

    Functions.pickle_save_object('/home/zhoul0a/Desktop/pulmonary_embolism/temp/cube_dict_list.pickle', cube_dict_list)
    exit()
    for index in range(150, 350, 10):
        Functions.merge_image_with_mask(
            np.clip(rescaled_ct_normal[:, :, index] + 0.5, 0, 1), mask_extracted[:, :, index])

    exit()
