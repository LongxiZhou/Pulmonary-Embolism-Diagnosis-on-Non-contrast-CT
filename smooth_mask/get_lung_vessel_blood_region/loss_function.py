import torch


def cross_entropy_get_blood_region(prediction, ground_truth, weight_array, class_balance_weight=(1, 1)):
    """

    prediction, ground_truth, weight_tensor in torch.FloatTensor format, on GPU,
    with shape [batch_size, class_num, x, y, z]
    here class_num is 2: they are, false blood region, other_voxel

    :param prediction: [batch_size, class_num, x, y, z], NOT soft_maxed!
    :param ground_truth: [batch_size, class_num, x, y, z], each pixel with value [0, 1]
    :param weight_array: [batch_size, class_num, x, y, z], each pixel with value [0, inf)
    :param class_balance_weight: balance_weight these classes
    :return: a float with value [0, inf)
    """

    softmax_then_log = torch.nn.LogSoftmax(dim=1)

    # calculate the loss
    loss_blood_refine = -softmax_then_log(prediction) * ground_truth * weight_array
    # add hyper balance for false blood region
    loss_blood_refine[:, 0] = loss_blood_refine[:, 0] * class_balance_weight[0]
    # add hyper balance for other region
    loss_blood_refine[:, 1] = loss_blood_refine[:, 1] * class_balance_weight[1]
    loss_blood_refine = torch.sum(loss_blood_refine)

    return loss_blood_refine
