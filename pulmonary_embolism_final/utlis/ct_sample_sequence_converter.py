"""
for each CT scan, it will be converted into a "sample sequence"
this .py file calculate the "sample_sequence"

sample sequence is form by many patch, which is extracted like convolution though the 3D CT array
we discard a patch if it does not contain blood vessel or only contain very tiny blood vessel

sample sequence is a list, and each element is a dict, recording information of a patch.
keys for the dict:
'ct_data': numpy_array, the patch extracted from CT
'depth_cube': numpy_array, the patch extracted from depth array of the blood vessel segmentation.
              More about depth_array for binary segmentation: for each voxel, outside segmentation is 0,
              inside segmentation is the distance from the nearest surface.
              depth_array = analysis.center_line_and_depth_3D.get_surface_distance(segmentation, strict=False)
'center_location': tuple like (100, 235, 192). the absolute location of the patch
'location_offset': tuple like (-5, 10, 2). the relative offset between this patch and the mass center of blood vessel
'branch_level': float, the average branch level of the vessels inside the patch
'penalty_weight_fp': numpy_array or None, determine voxel penalty for false positive classification.
                     None means penalty_weight_fp is all one
'penalty_weight_fn': numpy_array or None, determine voxel penalty for false negative classification.
                     None means penalty_weight_fn is all one
'clot_gt_mask': numpy_array or None. value from 0 to 1, 1 means this voxel should be classified as clot
'given_vector': numpy_array in shape (L, ) or None. If you want to fine-tune our model with prior-knowledge
                you can assign this value. In this study we set it to None


For training, each non-contrast CT will be convert to sample with format
{"sample_sequence": ,  # list

# the following keys determine sample class, loss function and simulation of clots
"center_line_loc_array": ,  # numpy loc array (return of np.where)
"is_PE": ,  # binary or, None (unknown class)
"has_clot_gt": ,  # binary or, None (not PE so clot gt is all zero)
"clot_gt_volume_sum":  # float or, None (no gt or clot gt is all zero)

"additional_information":  # dict or None
}

"""
import Tool_Functions.Functions as Functions
import numpy as np
import format_convert.spatial_normalize as spatial_normalize
import analysis.get_surface_rim_adjacent_mean as get_surface
from collaborators_package.denoise_chest_ct.denoise_predict import denoise_rescaled_array
import warnings


def convert_ct_into_tubes(rescaled_ct, depth_array=None, branch_array=None,
                          absolute_cube_length=(7, 7, 10), target_shape=(5, 5, 5),
                          resolution_rescaled_ct=(334/512, 334/512, 1),
                          max_cube_count=np.inf, min_depth=3.5, return_check=False, shift=(0, 0, 0),
                          step=None, clot_gt_mask=None, exclude_center_out=True, apply_denoise=False,
                          penalty_weight_fp=None, penalty_weight_fn=None, **semantic_dict):
    """
    the sample is extracted like convolution

    :param resolution_rescaled_ct: the length of each voxel in millimeters
    :param exclude_center_out: True to exclude cubes if its center is outside the valid_mask, i.e., vessel mask
                               False, all valid mask will be included in the sample_sequence
    :param clot_gt_mask:
    :param branch_array: branch_array gained from vessel mask
    :param depth_array: depth_array for the vessel segmentation
    :param rescaled_ct: in shape [512, 512, 512], each voxel with resolution [334/512, 334/512, 1] mm^3
    :param absolute_cube_length: the side length for the cube, in millimeters
    :param target_shape: the shape of the extracted cubes
    :param max_cube_count:
    :param min_depth: if the cube did not contain encoding_depth >= min_depth, it will be discarded, unless we cannot
    reach the num_cube_slice
    :param shift
    :param step: convolution step like (5, 5, 5) None means the step
    :param return_check: whether return a array same shape with rescaled_ct, indicate where we extract cubes.
    :param apply_denoise: if True, rescaled_ct = denoise_rescaled_array(rescaled_ct)
    :param penalty_weight_fp: numpy array, for training only
    :param penalty_weight_fn: numpy array, for training only
    :return: a list, each element is the return_dict of function "extract_cube"
    """

    assert np.shape(rescaled_ct) == np.shape(depth_array) and len(np.shape(rescaled_ct)) == 3

    if apply_denoise:
        rescaled_ct = denoise_rescaled_array(rescaled_ct)
    if depth_array is None or branch_array is None:
        depth_array, branch_array = get_depth_array_and_branch_array(rescaled_ct)

    cube_length = []

    for dim in range(3):
        if round(absolute_cube_length[dim] / resolution_rescaled_ct[dim]) % 2 == 0:
            cube_length.append(round(absolute_cube_length[dim] / resolution_rescaled_ct[dim]) + 1)
        else:
            cube_length.append(round(absolute_cube_length[dim] / resolution_rescaled_ct[dim]))

    print("cube_length:", cube_length)

    assert depth_array is not None

    location_array = np.where(depth_array > 3.5)

    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))

    print("mass center for blood vessel:", mass_center)

    max_depth = np.max(depth_array)
    print("max encoding_depth:", max_depth)

    if not return_check:
        sample_sequence = extract_vessel_sequence(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                                  max_cube_count, target_shape, return_check,
                                                  branch_array=branch_array,
                                                  clot_gt_mask=clot_gt_mask, penalty_weight_fp=penalty_weight_fp,
                                                  penalty_weight_fn=penalty_weight_fn,
                                                  exclude_center_out=exclude_center_out, min_depth=min_depth,
                                                  **semantic_dict)

        return sample_sequence

    sample_sequence, mask_check = \
        extract_vessel_sequence(rescaled_ct, cube_length, depth_array, mass_center, shift, step,
                                max_cube_count, target_shape, return_check, branch_array=branch_array,
                                clot_gt_mask=clot_gt_mask, penalty_weight_fp=penalty_weight_fp,
                                penalty_weight_fn=penalty_weight_fn,
                                exclude_center_out=exclude_center_out, min_depth=min_depth, **semantic_dict)

    return sample_sequence, mask_check


def get_depth_array_and_branch_array(rescaled_ct):
    import basic_tissue_prediction.predict_rescaled as predictor
    import analysis.center_line_and_depth_3D as get_center_line
    from pulmonary_embolism_v2.prepare_dataset.get_branch_mask import get_branching_cloud

    blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct)
    depth_array = get_center_line.get_surface_distance(blood_vessel_mask)
    blood_center_line = get_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)
    branch_array = get_branching_cloud(blood_center_line, depth_array, search_radius=5, smooth_radius=1,
                                       step=1, weight_half_decay=20, refine_radius=4)

    if np.sum(blood_center_line) < 4500:
        raise warnings.warn('Bad sample: blood center line less than 4500 voxel', np.sum(blood_center_line))

    max_depth = np.max(depth_array)
    if max_depth < 20:
        print("max encoding_depth is:", max_depth)
        warnings.warn("Bad sample: the max encoding_depth of the blood vessel is too small.")

    mask_sampling = np.array(depth_array > (max_depth - 5), 'int16')
    non_zero_loc = Functions.get_location_list(np.where(mask_sampling > 0.5))

    ct_value = []
    for loc in non_zero_loc:
        ct_value.append(rescaled_ct[loc])
    ct_value = np.array(ct_value)
    ct_value = ct_value * 1600 - 600

    blood_hu_average = np.mean(ct_value)
    blood_noise_average = np.mean(np.abs(ct_value - blood_hu_average))

    if not 0 < blood_hu_average < 100:
        warnings.warn("Bad sample: seems not non-contrast CT. Blood region average HU: %.2f" % blood_hu_average)

    if blood_noise_average > 150:
        warnings.warn("Bad sample: high noise scan, with noise %.2f HU" % blood_noise_average)

    return depth_array, branch_array


def extract_vessel_sequence(rescaled_ct, cube_length, depth_array, mass_center, shift=(0, 0, 0),
                            step=None, max_cube_count=np.inf, target_shape=(5, 5, 5), return_check=False,
                            branch_array=None, clot_gt_mask=None, exclude_center_out=True,
                            min_depth=0.5, penalty_weight_fp=None, penalty_weight_fn=None, **semantic_dict):
    """
    like 3D convolution to extract cubes that inside lung
    :param min_depth: when slicing cubes, the valid mask depth_array > min_depth
    :param exclude_center_out: True to exclude cubes if its center is outside the valid_mask, i.e., vessel mask
                               False, all blood vessel will be included in the sample_sequence
    :param clot_gt_mask:
    :param branch_array:
    :param mass_center:
    :param rescaled_ct:
    :param cube_length: (x_length, y_length, z_length)
    :param depth_array: depth array for blood vessel mask
    :param shift: shift when making the 3D grid
    :param step: by default it will let step=cube_length
    :param max_cube_count: the max cube extracted
    :param target_shape: the resized shape for the extracted cube
    :param return_check: whether return the extracted_count_mask
    :param penalty_weight_fp:
    :param penalty_weight_fn:
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

    valid_mask = np.array(depth_array >= min_depth, 'float32')
    if exclude_center_out:  # as only cube with center inside valid mask will included, we slightly extend the
        # valid mask
        valid_mask = valid_mask + get_surface.get_surface(valid_mask, outer=True, strict=False)

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

                central_location = (x, y, z)  # central_location for the cube

                if exclude_center_out:
                    if valid_mask[x, y, z] < 0.5:  # the central_location outside the vessel mask
                        continue
                else:
                    valid_overlap = valid_mask[
                                 central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                 central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                 central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]
                    if np.sum(valid_overlap) == 0:  # the cube has no overlap with valid mask
                        continue

                if num_sample > max_cube_count:
                    break

                central_location_offset = (central_location[0] - mass_center[0], central_location[1] - mass_center[1],
                                           central_location[2] - mass_center[2])

                sample = extract_cube(rescaled_ct, depth_array, cube_length, central_location,
                                      central_location_offset, target_shape, branch_array=branch_array,
                                      clot_gt_mask=clot_gt_mask, penalty_weight_fp=penalty_weight_fp,
                                      penalty_weight_fn=penalty_weight_fn, **semantic_dict)

                if sample['branch_level'] >= 11.9:
                    continue

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


def extract_cube(rescaled_ct, depth_array, cube_length, central_location, central_location_offset,
                 target_shape=None, branch_array=None, record_branch_array=False,
                 clot_gt_mask=None, penalty_weight_fp=None, penalty_weight_fn=None, **semantic_dict):
    """
    :param penalty_weight_fn:
    :param penalty_weight_fp:
    :param clot_gt_mask:
    :param record_branch_array:
    :param branch_array:
    :param depth_array: depth_array for vessel
    :param rescaled_ct: numpy float32 array with shape like (512, 512, 512)
    :param cube_length: an tuple of int, mod 2 == 1, like (11, 11, 7)
    :param central_location: the absolute location of the cube center, like (256, 325, 178)
    :param central_location_offset: the offset of the cube center to the vessel mass center, like (13, 17, 55)
    :param target_shape: rescale te cube to the target_shape
    :return: dict with key 'ct_data', 'penalty_weight', 'given_vector', 'location_offset'
    """
    assert cube_length[0] % 2 == 1 and cube_length[1] % 2 == 1 and cube_length[2] % 2 == 1 and min(cube_length) > 0
    assert branch_array is not None

    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    ct_cube = rescaled_ct[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                          central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                          central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    # depth cube for vessel mask
    depth_cube = depth_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                             central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                             central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    branch_cube = branch_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                               central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                               central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]

    if clot_gt_mask is not None:
        clot_gt_cube = clot_gt_mask[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                    central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                    central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]
    else:
        clot_gt_cube = None

    if penalty_weight_fp is not None:
        fp_cube = penalty_weight_fp[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                    central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                    central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]
    else:
        fp_cube = None

    if penalty_weight_fn is not None:
        fn_cube = penalty_weight_fn[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                    central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                    central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]
    else:
        fn_cube = None

    cube_list_other_semantic = []
    key_list_other_semantic = list(semantic_dict.keys())

    for key in key_list_other_semantic:
        semantic_array = semantic_dict[key]
        semantic_cube = semantic_array[central_location[0] - cube_radius_x: central_location[0] + cube_radius_x + 1,
                                       central_location[1] - cube_radius_y: central_location[1] + cube_radius_y + 1,
                                       central_location[2] - cube_radius_z: central_location[2] + cube_radius_z + 1]
        cube_list_other_semantic.append(semantic_cube)

    loc_list_non_zero = Functions.get_location_list(np.where(branch_cube > 0)) + Functions.get_location_list(
        np.where(branch_cube < 0))
    non_zero_count = 0
    branch_level_average = 0
    for loc in loc_list_non_zero:
        branch_level_average += branch_cube[loc]
        non_zero_count += 1
    if non_zero_count > 0:
        branch_level_average = branch_level_average / non_zero_count
    else:
        branch_level_average = 12  # we may delete this cube

    if target_shape is not None:
        # type ct_cube should in numpy.float32
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, target_shape, change_format=True)
        depth_cube = spatial_normalize.rescale_to_new_shape(depth_cube, target_shape, change_format=True)
        branch_cube = spatial_normalize.rescale_to_new_shape(branch_cube, target_shape, change_format=True)
        if clot_gt_cube is not None:
            clot_gt_cube = spatial_normalize.rescale_to_new_shape(clot_gt_cube, target_shape, change_format=True)
        if fp_cube is not None:
            fp_cube = spatial_normalize.rescale_to_new_shape(fp_cube, target_shape, change_format=True)
        if fn_cube is not None:
            fn_cube = spatial_normalize.rescale_to_new_shape(fn_cube, target_shape, change_format=True)
        for i in range(len(cube_list_other_semantic)):
            cube_list_other_semantic[i] = spatial_normalize.rescale_to_new_shape(
                cube_list_other_semantic[i], target_shape, change_format=True)

    ct_cube = np.array(ct_cube, 'float16')
    depth_cube = np.array(depth_cube, 'float16')
    branch_cube = np.array(branch_cube, 'float16')
    if clot_gt_cube is not None:
        clot_gt_cube = np.array(clot_gt_cube, 'float16')
    if fp_cube is not None:
        fp_cube = np.array(fp_cube, 'float16')
    if fn_cube is not None:
        fn_cube = np.array(fn_cube, 'float16')
    for i in range(len(cube_list_other_semantic)):
        cube_list_other_semantic[i] = np.array(cube_list_other_semantic[i], 'float16')

    return_dict = {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
                   'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube,
                   'branch_level': float(branch_level_average), 'clot_gt_mask': clot_gt_cube,
                   'penalty_weight_fp': fp_cube, 'penalty_weight_fn': fn_cube
                   }

    for i in range(len(cube_list_other_semantic)):
        return_dict[key_list_other_semantic[i]] = cube_list_other_semantic[i]

    if record_branch_array:
        return_dict['branch_array'] = branch_cube

    return return_dict


def reconstruct_semantic_from_sample_sequence(sample_sequence, absolute_cube_length=(4, 4, 5), show=False,
                                              key="ct_data", background=0.):
    """
    :param background: value for undefined region
    :param key: the semantic to reconstruct
    :param sample_sequence: list of dict, each like
    {'ct_data': ct_cube, 'penalty_weight': penalty_cube, 'location_offset': center_location_offset,
                   'given_vector': None, 'center_location': center_location, 'depth_cube': depth_cube}
    :param absolute_cube_length: (7, 7, 10) for non-high resolution, (4, 4, 5) for high resolution, (millimeters)
    :param show:
    :return: rescaled ct in shape [512, 512, 512]
    """
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

    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    if show:
        print("cube_length:", cube_length)

    assert cube_length[0] % 2 == 1 and cube_length[1] % 2 == 1 and cube_length[2] % 2 == 1 and min(cube_length) > 0

    rescaled_ct = np.zeros([512, 512, 512], 'float32')
    if not background == 0:
        rescaled_ct = rescaled_ct + background

    all_none = True

    for item in sample_sequence:
        if key not in item.keys():
            vale_array = None
        else:
            all_none = False
            vale_array = item[key]  # in shape target_shape, like (5, 5, 5)
        if type(vale_array) is int or type(vale_array) is float:
            vale_array = np.zeros(cube_length, 'float32') + vale_array
        elif vale_array is not None:
            vale_array = spatial_normalize.rescale_to_new_shape(np.array(vale_array, 'float32'), cube_length)
        else:
            vale_array = np.zeros(cube_length, 'float32')

        center_location = item["center_location"]

        rescaled_ct[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                    center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                    center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] = vale_array

    if np.sum(rescaled_ct) == 0:
        warnings.warn("the key %s has no value" % key)
    if all_none:
        raise KeyError("all item has not key %s" % key)

    return rescaled_ct


def example_convert_ct_to_sample_sequence_then_predict_clot():
    import pe_dataset_management.basic_functions as basic_functions
    import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    patient_id = 'patient-id-135'
    resolution = (4, 4, 5)
    denoise_ = False
    exclude_center_out = True
    min_depth = 2.5
    dataset_dir_cta, dataset_dir_non = basic_functions.find_patient_id_dataset_correspondence(patient_id, strip=True)
    if denoise_:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct-denoise', patient_id + '.npz'))['array']
    else:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct', patient_id + '.npz'))['array']
    depth_array = np.load(
        os.path.join(dataset_dir_non, 'depth_and_center-line/depth_array', patient_id + '.npz'))['array']
    branch_array = np.load(
        os.path.join(dataset_dir_non, 'depth_and_center-line/blood_branch_map', patient_id + '.npz'))['array']

    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, depth_array, branch_array, absolute_cube_length=resolution,
        exclude_center_out=exclude_center_out, min_depth=min_depth)

    print("length sample sequence:", len(sample_sequence))

    visualize_slice = 297

    extracted_patch_ = reconstruct_semantic_from_sample_sequence(
        sample_sequence, resolution, key='ct_data')

    Functions.image_show(np.clip(extracted_patch_[:, :, visualize_slice], -0.25, 0.7))

    sample_sequence_predicted = predict.predict_clot_for_sample_sequence(sample_sequence)

    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_predicted, resolution, key='clot_prob_mask')

    """
    Functions.merge_image_with_mask(
        np.clip(rescaled_ct[:, :, visualize_slice], -0.25, 0.7), predict_clot_mask[:, :, visualize_slice], show=True)
    """

    predict_clot_mask[0, 0, :] = 1
    predict_clot_mask[0, 1, :] = 0
    Functions.merge_image_with_mask(
        np.clip(rescaled_ct[:, :, visualize_slice], -0.25, 0.7), predict_clot_mask[:, :, visualize_slice], show=True)


if __name__ == '__main__':
    sample_ = Functions.pickle_load_object('/data_disk/pulmonary_embolism_final/'
                                           'samples_for_performance_evaluation/non_pe/high_resolution/'
                                           'not_pe_not_trim_not_denoise/trn08159.pickle')
    artery_mask_ = reconstruct_semantic_from_sample_sequence(sample_["sample_sequence"], key="artery_mask")
    import visualization.visualize_3d.visualize_stl as stl
    stl.visualize_numpy_as_stl(artery_mask_)
    exit()
    example_convert_ct_to_sample_sequence_then_predict_clot()
