"""
convert the format between rescaled_ct and sample_sequence_copy

function "slice_ct_to_sample_sequence" can generate the sample_sequence_copy for a target region
function "reconstruct_rescaled_ct_from_sample_sequence" can reconstruct the rescaled_ct from sample sequence


sample_sequence item dict:
{'ct_data': ct_cube, 'penalty_weight': penalty_cube, 'location_offset': center_location_offset,
                   'given_vector': None, 'center_location': center_location, 'depth_cube': depth_cube}

"""

import numpy as np
from pulmonary_embolism_v3.prepare_training_dataset.convert_ct_to_sample import convert_ct_into_tubes
import analysis.center_line_and_depth_3D as get_center_line
import basic_tissue_prediction.predict_rescaled as predictor
from pulmonary_embolism_v2.prepare_dataset.get_branch_mask import get_branching_cloud
from collaborators_package.denoise_chest_ct.denoise_predict import denoise_rescaled_array
import format_convert.spatial_normalize as spatial_normalize
import os
from pulmonary_embolism_v3.prepare_training_dataset.trim_refine_and_remove_bad_scan import reduce_sequence_length


def extract_sequence_from_rescaled_ct(rescaled_ct, blood_vessel_mask=None, depth_array=None,
                                      blood_center_line=None, branch_array=None, apply_denoise=False, strict_trim=False,
                                      high_resolution=False):

    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
    else:
        absolute_cube_length = (4, 4, 5)

    if apply_denoise:
        rescaled_ct = denoise_rescaled_array(rescaled_ct)

    if depth_array is None:
        if blood_vessel_mask is None:
            blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct)
        depth_array = get_center_line.get_surface_distance(blood_vessel_mask)

    if branch_array is None:
        if blood_center_line is None:
            if blood_vessel_mask is None:
                blood_vessel_mask = np.array(depth_array > 0.5, 'uint8')
            blood_center_line = get_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)
        branch_array = get_branching_cloud(blood_center_line, depth_array, search_radius=5, smooth_radius=1,
                                           step=1, weight_half_decay=20, refine_radius=4)

    sample_sequence = convert_ct_into_tubes(rescaled_ct, depth_array, branch_array, None,
                                            None, absolute_cube_length, only_v1=True,
                                            clot_gt_mask=None, exclude_center_out=False)

    if strict_trim:
        if high_resolution:
            target_length = 3000
        else:
            target_length = 1500
        max_branch = 7
    else:
        if high_resolution:
            target_length = 4000
        else:
            target_length = 3000
        max_branch = 9

    sample_sequence = reduce_sequence_length(sample_sequence, target_length=target_length, max_branch=max_branch)

    return sample_sequence


def reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, absolute_cube_length=(7, 7, 10), show=False,
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
    from segment_clot_cta.inference.inference_on_standard_dataset import predict_and_show
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided
    fn_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_ct-denoise')
    model = load_saved_model_guided(high_resolution=False)
    for fn in fn_list[0::2]:
        print("processing:", fn)
        predict_and_show(high_resolution=False, file_name=fn,
                         dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/',
                         image_save_dict='/data_disk/temp/visualize/clot_predict_no_gt/',
                         model_loaded=model,
                         save_dict_clot_mask='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                                             'rescaled_clot_predict')
    exit()
