import torch
import numpy as np
from med_transformer.position_embeding import get_4d_sincos_pos_embed_loc_list


def default_penalty_normalize_func(clot_volume_array, power_factor=0.5, relative_ratio=0.5):
    """

    total penalty for clot of volume V is: V ** (1 - power_factor) * relative_ratio

    :param relative_ratio:
    :param power_factor:
    :param clot_volume_array:
    :return: False_Negative penalty array
    """

    positive_mask = np.array(clot_volume_array > 0.5, 'float32')
    if np.sum(positive_mask) == 0:
        return positive_mask
    penalty_array = positive_mask / (np.power(clot_volume_array + 0.1, power_factor) / relative_ratio)
    return penalty_array


def prepare_tensors_simulate_clot(list_sample_sequence, embed_dim, device='cuda:0', training_phase=True,
                                  penalty_normalize_func=None, roi='blood_vessel', trace_clot=True,
                                  sample_sequence_len=None):
    """
    prepare batch_tensor, pos_embed_tensor, given_features, flatten_roi, cube_shape, clot_gt_tensor

    :param sample_sequence_len:
    :param list_sample_sequence: a list, length is batch_size, each item is a sample sequence. Each sample sequence is
    a list, the item is a dict:
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'branch_level': 'branch_level',
    'clot_array': None}

    :param training_phase: during training, calculate the clot_gt_tensor
    :param embed_dim: int
    :param device: None to return numpy arrays
    :param penalty_normalize_func, apply on numpy array.
    penalty for False_Negative = penalty_normalize_func(clot_volume_array)
    :param roi: region of interest during segmentation
    :param trace_clot: if True, build the clot_volume_vectors
    :return: batch_tensor, pos_embed_tensor, given_vector, flatten_roi, cube_shape, clot_gt_tensor
    """
    batch_size = len(list_sample_sequence)
    assert batch_size > 0
    if penalty_normalize_func is None:
        penalty_normalize_func = default_penalty_normalize_func

    example_sample = list_sample_sequence[0][0]

    if sample_sequence_len is None:
        sample_sequence_len = 0
        for sample_sequence in list_sample_sequence:
            if len(sample_sequence) > sample_sequence_len:
                sample_sequence_len = len(sample_sequence)
    else:
        assert type(sample_sequence_len) is int
        for sample_sequence in list_sample_sequence:
            assert sample_sequence_len >= len(sample_sequence)

    cube_shape = np.shape(example_sample['ct_data'])
    batch_array = np.zeros([batch_size, 1, cube_shape[0], cube_shape[1],
                            cube_shape[2] * sample_sequence_len], 'float32')
    if example_sample['given_vector'] is None:
        given_vector_array = None
        given_dim = 0
    else:
        given_vector_array = np.zeros(
            [batch_size, sample_sequence_len, len(example_sample['given_vector'])], 'float32')
        given_dim = len(example_sample['given_vector'])

    flatten_roi_region = np.zeros(
        [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')

    if training_phase:
        gt_vectors = np.zeros(
            [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')
        if trace_clot:
            clot_volume_vectors = np.zeros(
                [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')
        else:
            clot_volume_vectors = None
    else:
        gt_vectors = None
        clot_volume_vectors = None

    location_list = []

    # complete these arrays
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            item = list_sample_sequence[i][j]
            batch_array[i, 0, :, :, j * cube_shape[2]: (j + 1) * cube_shape[2]] = item['ct_data']
            x_c, y_c, z_c = item['location_offset']
            branch_level = item['branch_level']
            location_list.append((x_c, y_c, z_c, branch_level))
            if given_dim > 0:
                given_vector_array[i, j, :] = item['given_vector']

            if roi == 'blood_region':
                blood_region = item["blood_region"] * np.array(
                    item['depth_cube'] > max(0.5, 6 - item['branch_level']), 'float32')
                flatten_roi_region[i, j, :] = np.reshape(blood_region, (-1,))
            else:
                assert roi == 'blood_vessel'
                vessel_region = np.array(item['depth_cube'] > 0.5, 'float32')
                flatten_roi_region[i, j, :] = np.reshape(vessel_region, (-1,))

            if training_phase:
                if 'clot_array' not in item.keys():
                    item['clot_array'] = None
                if item['clot_array'] is not None:  # not 0 means this is a simulated clot
                    clot_mask_array = \
                        np.array(item['clot_array'] > 0, 'float32') + np.array(item['clot_array'] < 0, 'float32')
                    gt_vectors[i, j, :] = np.reshape(clot_mask_array, (-1,))
                if trace_clot:
                    if 'clot_volume_array' in item.keys():
                        if item['clot_volume_array'] is not None:  # greater than 0 means this is a simulated clot
                            clot_volume_vectors[i, j, :] = np.reshape(
                                np.array(item['clot_volume_array'], 'float32'), (-1,))

    # get positional encoding array
    pos_embed_array_temp = get_4d_sincos_pos_embed_loc_list(embed_dim, location_list)  # [len(loc_list), embed_dim]
    pos_embed_array = np.zeros([batch_size, sample_sequence_len, embed_dim], 'float32')
    shift = 0
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            pos_embed_array[i, j, :] = pos_embed_array_temp[shift, :]
            shift += 1

    array_packages = batch_array, pos_embed_array, given_vector_array, flatten_roi_region, clot_volume_vectors, \
        gt_vectors, cube_shape

    if device is None:
        return array_packages

    else:
        return put_arrays_on_device_simu_clot(array_packages, device, training_phase, penalty_normalize_func)


def put_arrays_on_device_simu_clot(array_packages, device='cuda:0', training_phase=True, penalty_normalize_func=None,
                                   trace_clot=True):
    if array_packages is None:
        return None
    batch_array, pos_embed_array, given_vector_array, flatten_roi, clot_volume_vectors, \
        gt_vectors, cube_shape = array_packages

    if penalty_normalize_func is None:
        penalty_normalize_func = default_penalty_normalize_func

    # form torch tensors
    batch_tensor = torch.FloatTensor(batch_array).cuda(device)
    pos_embed_tensor = torch.FloatTensor(pos_embed_array).cuda(device)
    if given_vector_array is not None:
        given_vector = torch.FloatTensor(given_vector_array).cuda(device)
    else:
        given_vector = None

    flatten_roi = torch.FloatTensor(flatten_roi).cuda(device)

    if training_phase:
        gt_vectors_negative = np.array(1 - gt_vectors, 'float32')
        clot_gt_tensor_positive = torch.FloatTensor(gt_vectors).cuda(device)
        clot_gt_tensor_negative = torch.FloatTensor(gt_vectors_negative).cuda(device)

        if trace_clot:
            penalty_weight_fp = torch.FloatTensor(gt_vectors_negative).cuda(device)
            penalty_weight_fn = torch.FloatTensor(penalty_normalize_func(clot_volume_vectors)).cuda(device)
            penalty_weight_tensor = torch.stack((penalty_weight_fp, penalty_weight_fn), dim=1)
            # [B, 2, N, flatten_dim]
        else:
            penalty_weight_tensor = None

        clot_gt_tensor = torch.stack((clot_gt_tensor_negative, clot_gt_tensor_positive), dim=1)
        # [B, 2, N, flatten_dim]

    else:
        clot_gt_tensor = None
        penalty_weight_tensor = None

    return batch_tensor, pos_embed_tensor, given_vector, \
        flatten_roi, cube_shape, clot_gt_tensor, penalty_weight_tensor


if __name__ == '__main__':
    exit()
