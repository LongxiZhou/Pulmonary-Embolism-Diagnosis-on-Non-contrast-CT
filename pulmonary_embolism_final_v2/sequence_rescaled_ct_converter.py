"""
convert the format between rescaled_ct and sample_sequence_copy

function "slice_ct_to_sample_sequence" can generate the sample_sequence_copy for a target region
function "reconstruct_rescaled_ct_from_sample_sequence" can reconstruct the rescaled_ct from sample sequence


sample_sequence item dict:
{'ct_data': ct_cube, 'penalty_weight': penalty_cube, 'location_offset': center_location_offset,
                   'given_vector': None, 'center_location': center_location, 'depth_cube': depth_cube}

"""

import numpy as np
from pulmonary_embolism_v2.prepare_dataset.convert_blood_vessel_to_sliced_sequence import convert_ct_into_tubes
import analysis.center_line_and_depth_3D as get_center_line
import basic_tissue_prediction.predict_rescaled as predictor
from pulmonary_embolism_v2.prepare_dataset.get_branch_mask import get_branching_cloud
# from collaborators_package.artery_vein_segmentation.predict import predict_artery_and_vein
from collaborators_package.denoise_chest_ct.denoise_predict import denoise_rescaled_array
import format_convert.spatial_normalize as spatial_normalize
import pulmonary_embolism_v2.sequence_operations.trim_length as trim_length
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def extract_sequence_from_rescaled_ct(rescaled_ct, blood_vessel_mask=None,
                                      blood_center_line=None, branch_array=None, apply_denoise=True, trim=True):

    if apply_denoise:
        rescaled_ct = denoise_rescaled_array(rescaled_ct)

    if blood_vessel_mask is None:
        blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct)

    depth_array = get_center_line.get_surface_distance(blood_vessel_mask)

    if blood_center_line is None:
        blood_center_line = get_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)

    if branch_array is None:
        branch_array = get_branching_cloud(blood_center_line, depth_array, search_radius=5, smooth_radius=1,
                                           step=1, weight_half_decay=20, refine_radius=4)

    sample_sequence = convert_ct_into_tubes(rescaled_ct, blood_vessel_mask, absolute_cube_length=(4, 4, 5),
                                            target_shape=(5, 5, 5), max_cube_count=np.inf, min_depth=4, show=True,
                                            return_check=False, shift=(0, 0, 0),
                                            step=None, only_v1=True, only_v2=False, depth_array=depth_array,
                                            mass_center=None,
                                            branch_array=branch_array)

    if trim:
        sample_sequence = trim_length.reduce_sequence_length(sample_sequence)

    return sample_sequence


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
        if key not in item.keys():
            vale_array = None
        else:
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

    return rescaled_ct


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    sequence_sample = Functions.pickle_load_object(
        '/data_disk/pulmonary_embolism/training_dataset_no_clip_no_center_line/combine_ready_not_denoise/'
        'trn03750.pickle')["sample_sequence"]

    print("length of sequence:", len(sequence_sample))

    array_reconstruct = reconstruct_rescaled_ct_from_sample_sequence(sequence_sample, (4, 4, 5), key='ct_data')
    for z in range(250, 350, 5):
        Functions.image_show(array_reconstruct[:, :, z])
    exit()
