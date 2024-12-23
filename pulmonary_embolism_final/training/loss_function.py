import torch
import numpy as np


def weighted_cross_entropy_loss(segmentation_before_softmax, gt_tensor, class_balance_weights=None,
                                sample_balance_weights=None, voxel_penalty_tensor=None):
    """

    :param voxel_penalty_tensor: tensor in shape [B, class, ...],
                                record the false prediction penalty for each class each voxel
    :param sample_balance_weights:
    :param segmentation_before_softmax: tensor in shape [B, class, ...],
    :param gt_tensor: tensor in shape [B, class, ...], satisfies torch.sum(gt_tensor, dim=1) -> all_file elements are 1.
    :param class_balance_weights: false prediction penalty for each class
    :return: the loss
    """
    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_for_prediction_probability = -softmax_then_log(segmentation_before_softmax)

    class_num = gt_tensor.shape[1]
    sample_num = gt_tensor.shape[0]

    if class_balance_weights is not None:
        for i in range(class_num):
            log_for_prediction_probability[:, i] = class_balance_weights[i] * log_for_prediction_probability[:, i]

    if sample_balance_weights is not None:
        for j in range(sample_num):
            log_for_prediction_probability[j] = sample_balance_weights[j] * log_for_prediction_probability[j]

    if voxel_penalty_tensor is not None:
        log_for_prediction_probability = log_for_prediction_probability * voxel_penalty_tensor

    return_tensor = log_for_prediction_probability * gt_tensor
    return_tensor = torch.sum(return_tensor)

    return return_tensor


def form_tensors(batch_sample, penalty_for_padding_cubes=None):
    """
    :param batch_sample: {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
    "list_penalty_array_sequence":}, is the iteration output of the "DataLoaderForPE"
    :param penalty_for_padding_cubes: None, or a numpy float32 cube in shape [X, Y, Z], indication penalty for padding
    cubes. None for do not give any penalty.

    :return: two tensor in shape [batch_size, num_query_cubes, 1, X, Y, Z], on CPU, one for gt, one for penalty weights
    """

    list_information_sequence = batch_sample["list_sample_sequence"]
    list_query_sequence = batch_sample["list_query_sequence"]
    list_ct_data_sequence = batch_sample["list_ct_data_sequence"]
    list_penalty_array_sequence = batch_sample["list_penalty_array_sequence"]

    batch_size = len(list_information_sequence)
    num_query_cubes = 0
    for i in range(batch_size):
        if len(list_query_sequence[i]) > num_query_cubes:
            num_query_cubes = len(list_query_sequence[i])

    x, y, z = np.shape(list_ct_data_sequence[0][0])

    temp_array_gt = np.zeros([batch_size, num_query_cubes, 1, x, y, z], 'float32')
    temp_array_penalty = np.zeros([batch_size, num_query_cubes, 1, x, y, z], 'float32')

    for i in range(batch_size):
        for j in range(len(list_ct_data_sequence[i])):
            temp_array_gt[i, j, 0, :, :, :] = list_ct_data_sequence[i][j]
            temp_array_penalty[i, j, 0, :, :, :] = list_penalty_array_sequence[i][j]

    if penalty_for_padding_cubes is not None:
        for i in range(batch_size):
            for j in range(len(list_ct_data_sequence[i]), num_query_cubes):
                temp_array_penalty[i, j, 0, :, :, :] = penalty_for_padding_cubes

    tensor_gt = torch.FloatTensor(temp_array_gt)
    tensor_penalty = torch.FloatTensor(temp_array_penalty)

    return tensor_gt, tensor_penalty


def form_tensors_tissue_wise(batch_sample, penalty_for_padding_cubes=None):
    """
    :param batch_sample: {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
    "list_penalty_array_sequence":}, is the iteration output of the "DataLoaderForPE"
    :param penalty_for_padding_cubes: None, or a numpy float32 cube in shape [X, Y, Z], indication penalty for padding
    cubes. None for give average penalty of other tissue.

    :return: two tensor in shape [batch_size, num_query_cubes, 1, X, Y, Z], on CPU, one for gt, one for penalty weights
    """

    list_information_sequence = batch_sample["list_sample_sequence"]
    list_query_sequence = batch_sample["list_query_sequence"]
    list_ct_data_sequence = batch_sample["list_ct_data_sequence"]
    list_penalty_array_sequence = batch_sample["list_penalty_array_sequence"]

    batch_size = len(list_information_sequence)
    num_query_cubes = 0
    for i in range(batch_size):
        if len(list_query_sequence[i]) > num_query_cubes:
            num_query_cubes = len(list_query_sequence[i])

    x, y, z = np.shape(list_ct_data_sequence[0][0])

    temp_array_gt = np.zeros([batch_size, num_query_cubes, 1, x, y, z], 'float32')
    temp_array_penalty = np.zeros([batch_size, num_query_cubes, 1, x, y, z, 4], 'float32')

    for i in range(batch_size):
        for j in range(len(list_ct_data_sequence[i])):
            temp_array_gt[i, j, 0, :, :, :] = list_ct_data_sequence[i][j]
            temp_array_penalty[i, j, 0, :, :, :] = list_penalty_array_sequence[i][j]

    if penalty_for_padding_cubes is None:
        mean_penalty = np.average(temp_array_penalty)
        penalty_for_padding_cubes = mean_penalty

    for i in range(batch_size):
        for j in range(len(list_ct_data_sequence[i]), num_query_cubes):
            temp_array_penalty[i, j, 0, :, :, :] = penalty_for_padding_cubes

    tensor_gt = torch.FloatTensor(temp_array_gt)
    tensor_penalty = torch.FloatTensor(temp_array_penalty)

    return tensor_gt, tensor_penalty


def form_tensors_tissue_wise_v2(batch_sample, penalty_for_padding_cubes=None):
    """

    compared to "form_tensors_tissue_wise", this version require the model_guided to predict information cubes

    :param batch_sample: {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
    "list_penalty_array_sequence":}, is the iteration output of the "DataLoaderForPE"
    :param penalty_for_padding_cubes: None, or a numpy float32 cube in shape [X, Y, Z], indication penalty for padding
    cubes. None for do not give any penalty.

    :return: two tensor on CPU,
    one for gt, one for penalty weights
    for gt, in shape in shape [batch_size, num_information_cubes + num_query_cubes, 1, X, Y, Z],
    for penalty weights in shape [batch_size, num_information_cubes + num_query_cubes, 1, X, Y, Z, 4]
    """

    list_information_sequence = batch_sample["list_sample_sequence"]
    list_query_sequence = batch_sample["list_query_sequence"]
    list_ct_data_sequence = batch_sample["list_ct_data_sequence"]
    list_penalty_array_sequence = batch_sample["list_penalty_array_sequence"]

    batch_size = len(list_information_sequence)
    num_query_cubes = 0
    num_information_cubes = 0

    for i in range(batch_size):
        if len(list_query_sequence[i]) > num_query_cubes:
            num_query_cubes = len(list_query_sequence[i])
        if len(list_information_sequence[i]) > num_information_cubes:
            num_information_cubes = len(list_information_sequence[i])

    x, y, z = np.shape(list_ct_data_sequence[0][0])

    temp_array_gt = np.zeros([batch_size, num_information_cubes + num_query_cubes, 1, x, y, z], 'float32')
    temp_array_penalty = np.zeros([batch_size, num_information_cubes + num_query_cubes, 1, x, y, z, 4], 'float32')

    for i in range(batch_size):
        for j in range(len(list_information_sequence[i])):
            temp_array_gt[i, j, 0, :, :, :] = list_information_sequence[i][j]["ct_data"]
            temp_array_penalty[i, j, 0, :, :, :] = list_information_sequence[i][j]["penalty_weight"]

        for j in range(len(list_ct_data_sequence[i])):
            temp_array_gt[i, num_information_cubes + j, 0, :, :, :] = list_ct_data_sequence[i][j]
            temp_array_penalty[i, num_information_cubes + j, 0, :, :, :] = list_penalty_array_sequence[i][j]

    if penalty_for_padding_cubes is None:
        penalty_for_padding_cubes = np.average(temp_array_penalty)

    for i in range(batch_size):
        for j in range(len(list_information_sequence[i]), num_information_cubes):
            temp_array_penalty[i, j, 0, :, :, :] = penalty_for_padding_cubes
        for j in range(len(list_ct_data_sequence[i]), num_query_cubes):
            temp_array_penalty[i, num_information_cubes + j, 0, :, :, :] = penalty_for_padding_cubes

    tensor_gt = torch.FloatTensor(temp_array_gt)
    tensor_penalty = torch.FloatTensor(temp_array_penalty)

    return tensor_gt, tensor_penalty


def form_tensors_tissue_wise_v3(batch_sample, global_penalty_weight=(1, 0.1), training=True):
    """

    "global_penalty_weight" controls the relative pixel-wise penalty for padding and non-padding cubes.

    compared to "form_tensors_tissue_wise", this version require the model_guided to predict information cubes
    compared to "form_tensors_tissue_wise_v2", this version can change global penalty for padding and non-padding
    separately, but all_file voxel penalty for non-padding/padding is the same

    :param training:
    :param batch_sample: {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
    "list_penalty_array_sequence":}, is the iteration output of the "DataLoaderForPE"
    :param global_penalty_weight: (voxel_penalty_non_padding, voxel_penalty_padding)

    :return: two tensor on CPU,
    one for gt, one for penalty weights
    for gt, in shape in shape [batch_size, num_information_cubes + num_query_cubes, 1, X, Y, Z],
    for penalty weights in shape [batch_size, num_information_cubes + num_query_cubes, 1, X, Y, Z, 4]
    """

    list_information_sequence = batch_sample["list_sample_sequence"]
    if "list_ct_data_sequence" in list(batch_sample.keys()):
        list_ct_data_sequence = batch_sample["list_ct_data_sequence"]
    else:
        list_ct_data_sequence = []
        list_query_gt_sequence = batch_sample["list_query_gt_sequence"]
        for query_gt_sequence in list_query_gt_sequence:
            ct_data_sequence = []
            for item in query_gt_sequence:
                ct_data_sequence.append(item['ct_data'])
            list_ct_data_sequence.append(ct_data_sequence)

    batch_size = len(list_information_sequence)
    num_query_cubes = 0
    num_information_cubes = 0

    for i in range(batch_size):
        if len(list_ct_data_sequence[i]) > num_query_cubes:
            num_query_cubes = len(list_ct_data_sequence[i])
        if len(list_information_sequence[i]) > num_information_cubes:
            num_information_cubes = len(list_information_sequence[i])

    x, y, z = np.shape(list_ct_data_sequence[0][0])

    temp_array_gt = np.zeros([batch_size, num_information_cubes + num_query_cubes, 1, x, y, z], 'float32')
    temp_array_penalty = np.zeros([batch_size, num_information_cubes + num_query_cubes, 1, x, y, z, 4], 'float32')

    for i in range(batch_size):  # complete the non-padding part
        for j in range(len(list_information_sequence[i])):
            temp_array_gt[i, j, 0, :, :, :] = list_information_sequence[i][j]["ct_data"]
            temp_array_penalty[i, j, 0, :, :, :] = global_penalty_weight[0]

        for j in range(len(list_ct_data_sequence[i])):
            temp_array_gt[i, num_information_cubes + j, 0, :, :, :] = list_ct_data_sequence[i][j]
            temp_array_penalty[i, num_information_cubes + j, 0, :, :, :] = global_penalty_weight[0]

    for i in range(batch_size):  # complete the padding part
        for j in range(len(list_information_sequence[i]), num_information_cubes):
            temp_array_penalty[i, j, 0, :, :, :] = global_penalty_weight[1]
        for j in range(len(list_ct_data_sequence[i]), num_query_cubes):
            temp_array_penalty[i, num_information_cubes + j, 0, :, :, :] = global_penalty_weight[1]

    tensor_gt = torch.FloatTensor(temp_array_gt)
    tensor_penalty = torch.FloatTensor(temp_array_penalty)

    if training:
        return tensor_gt, tensor_penalty
    return tensor_gt[:, num_information_cubes::], tensor_penalty[:, num_information_cubes::, ]


if __name__ == '__main__':
    exit()
