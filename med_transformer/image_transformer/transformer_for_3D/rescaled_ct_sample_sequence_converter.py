"""
convert the format between rescaled_ct and sample_sequence_copy

function "slice_ct_to_sample_sequence" can generate the sample_sequence_copy for a target region
function "reconstruct_rescaled_ct_from_sample_sequence" can reconstruct the rescaled_ct from sample sequence

"""

import numpy as np
import med_transformer.image_transformer.transformer_for_3D.convert_ct_to_sliced_sequence as slicer
import analysis.center_line_and_depth_3D as get_center_line
import basic_tissue_prediction.predict_rescaled as predictor
# from collaborators_package.artery_vein_segmentation.predict import predict_artery_and_vein
import format_convert.spatial_normalize as spatial_normalize


def slice_ct_to_sample_sequence(rescaled_ct, focal_region=None, vessel_depth_mask=None, absolute_cube_length=(7, 7, 10),
                                target_shape=(5, 5, 5), max_cube_count=np.inf, min_depth=3, show=True,
                                return_check=False, shift=(0, 0, 0), step=None, mass_center=None, bounding_box=None):
    """

    :param rescaled_ct:
    :param focal_region: binary mask for where to slice tubes, None for all_file region
    :param vessel_depth_mask: for determine the relative location
    :param absolute_cube_length: in mm
    :param target_shape: shape of each cube
    :param max_cube_count:
    :param min_depth: when determine the mass center of blood vessel, remove surface voxels to be more robust
    :param show:
    :param return_check: if True, further return the slice count mask
    :param shift: a tuple, add global, apply on the 512 by 512 by 512 array.
    :param step: step during slicing, like step in 3D convolution. default as the cube size
    :param mass_center: mass_center of the vessel
    :param bounding_box: bounding box for the focal semantic
    :return: list of dict, each like:
    {'ct_data': ct_cube, 'penalty_weight': penalty_cube, 'location_offset': center_location_offset,
                   'given_vector': None, 'center_location': center_location, 'depth_cube': depth_cube}
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

    if show:
        print("cube_length:", cube_length)

    if mass_center is None:
        if vessel_depth_mask is None:
            vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct)

            vessel_depth_mask = get_center_line.get_surface_distance(vessel_mask)

        assert min_depth >= 1
        location_array = np.where(vessel_depth_mask > (min_depth - 0.5))

        mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                       int(np.average(location_array[2])))
    if show:
        print("mass center:", mass_center)

    if focal_region is None:
        assert bounding_box is None
        return slicer.pipeline_extract_all(rescaled_ct, None, cube_length, mass_center, shift, step, max_cube_count,
                                           target_shape, return_check)

    return slicer.pipeline_extract_mask(rescaled_ct, None, cube_length, focal_region, mass_center, shift, step,
                                        max_cube_count, target_shape, return_check, bounding_box=bounding_box)


def reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, absolute_cube_length, show=False,
                                                 key="ct_data"):
    """

    :param key: the data to reconstruct
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

    for item in sample_sequence:
        ct_cube = item[key]  # in shape target_shape, like (5, 5, 5)
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, cube_length)
        center_location = item["center_location"]

        rescaled_ct[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                    center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                    center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] = ct_cube

    return rescaled_ct


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions
    sequence_sample = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset/normal_scan_extended/v1/xwzc000234_unknown.pickle')

    array_reconstruct = reconstruct_rescaled_ct_from_sample_sequence(sequence_sample, (7, 7, 10))
    for z in range(250, 350, 5):
        Functions.image_show(array_reconstruct[:, :, z])
    exit()
