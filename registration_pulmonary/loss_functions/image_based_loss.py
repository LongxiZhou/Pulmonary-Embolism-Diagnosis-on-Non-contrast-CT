"""
loss functions based on the registered image and fixed images
"""


import math
import torch
import numpy as np
import torch.nn.functional as F


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def mae_loss(x, y):
    return torch.mean(torch.abs(x - y))


def mae_loss_normalized(x, y):  # same with dice loss when binary
    return torch.sum(torch.abs(x - y)) / (torch.sum(torch.abs(x)) + torch.sum(torch.abs(y)))


def mse_loss_normalized(x, y):  # same with dice loss when binary
    return torch.sum(torch.square(x - y)) / (torch.sum(torch.square(x)) + torch.sum(torch.square(y)))


def mae_pair(registered, fixed):
    return torch.abs(registered - fixed)


def mse_pair(registered, fixed):
    return torch.square(registered - fixed)


def voxel_pair_loss_on_focal_region(registered, fixed, penalty_weights_pair, loss_func_voxel_pair):
    """

    :param loss_func_voxel_pair: like mae_pair, mse_pair
    [Batch, C, X, Y, Z] = loss_func_voxel_pair(registered, fixed)

    :param registered: torch float tensor with shape [Batch, C, X, Y, Z]
    :param fixed: torch float tensor with shape [Batch, C, X, Y, Z]
    :param penalty_weights_pair: torch float tensor with shape [Batch, C, X, Y, Z]
    :return:
    """

    voxel_loss_pair = loss_func_voxel_pair(registered, fixed)

    weighted_voxel_loss = voxel_loss_pair * penalty_weights_pair

    return torch.mean(weighted_voxel_loss)


def dice_loss_fn(moving_mask, fixed_mask):
    intersect = torch.sum(moving_mask * fixed_mask)
    denominator = torch.sum(moving_mask * moving_mask) + torch.sum(fixed_mask * fixed_mask)
    return 1 - 2 * (intersect / denominator)


def weighted_ncc_loss(registered_image, fixed_image, weight_map=None,
                      win_length=9, stride_step=1, show=False, to_cuda=True, return_cc_map=False):
    """

    :param registered_image: torch float tensor in shape [N, 1, X, Y, Z]
    :param fixed_image: torch float tensor in shape [N, 1, X, Y, Z]
    :param weight_map: None or torch float tensor in shape [N, 1, X, Y, Z], should be non negative
    :param win_length:
    :param stride_step:
    :param show:
    :param to_cuda:
    :param return_cc_map
    :return: ncc loss in range [-1, 0]
    """
    dim = 3
    win = [win_length] * dim
    sum_filter = torch.ones([1, 1, *win])
    if to_cuda:
        sum_filter = sum_filter.to("cuda:0")
    pad_no = math.floor(win[0] / 2)
    stride = [stride_step] * dim
    padding = [pad_no] * dim

    var_r, var_f, cross, mean_w = compute_weighted_local_sums(
        registered_image, fixed_image, sum_filter, stride, padding, win, weight_map=weight_map)

    # cc = (cross * cross + 1e-5) / (var_f * var_r + 1e-5)  # shape cc is same shape with input tensor
    cc = (cross * cross) / (var_f * var_r + 1e-5)  # shape cc is same shape with input tensor
    if mean_w is not None:
        cc = mean_w * cc

    if show:
        numpy_cc = cc.detach().cpu().numpy()[0, 0]
        Functions.image_show(numpy_cc[:, :, int(np.shape(numpy_cc)[2] / 2)])

    if return_cc_map:
        return cc

    return -1 * torch.mean(cc)


def compute_weighted_local_sums(registered_image, fixed_image, sum_kernel, stride, padding, win, weight_map=None):
    r_sum = F.conv3d(registered_image, sum_kernel, stride=stride, padding=padding)
    f_sum = F.conv3d(fixed_image, sum_kernel, stride=stride, padding=padding)

    r2_sum = F.conv3d(registered_image * registered_image, sum_kernel, stride=stride, padding=padding)
    f2_sum = F.conv3d(fixed_image * fixed_image, sum_kernel, stride=stride, padding=padding)
    rf_sum = F.conv3d(registered_image * fixed_image, sum_kernel, stride=stride, padding=padding)

    win_size = np.prod(win)

    mean_r = r_sum / win_size
    mean_f = f_sum / win_size
    cross = rf_sum - mean_f * r_sum - mean_r * f_sum + mean_r * mean_f * win_size

    var_r = r2_sum - 2 * mean_r * r_sum + mean_r * mean_r * win_size
    var_f = f2_sum - 2 * mean_f * f_sum + mean_f * mean_f * win_size

    if weight_map is not None:
        w_sum = F.conv3d(weight_map, sum_kernel, stride=stride, padding=padding)
        mean_w = w_sum / win_size
    else:
        mean_w = None

    return var_r, var_f, cross, mean_w


def ncc_loss(I, J, win=None, stride_step=1, show=False):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:0")
    pad_no = math.floor(win[0] / 2)
    stride = [stride_step] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = (cross * cross + 1e-5) / (I_var * J_var + 1e-5)  # shape cc is same shape with input tensor

    if show:
        import Tool_Functions.Functions as Functions
        numpy_cc = cc.detach().cpu().numpy()[0, 0]
        Functions.image_show(numpy_cc[:, :, int(np.shape(numpy_cc)[2] / 2)])

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


if __name__ == '__main__':

    import Tool_Functions.Functions as Functions
    from Tool_Functions.performance_metrics import dice_score_two_class
    from format_convert.basic_transformations import down_sample_central_mass_center_and_crop_size, transformation_on_array

    vessel_cta = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/simulated_non_contrast/semantics/blood_mask/patient-id-24534412.npz')[
        'array']
    vessel_non = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality/semantics/blood_mask/patient-id-24534412.npz')[
        'array']

    depth_cta = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/simulated_non_contrast/depth_and_center-line/depth_array/patient-id-24534412.npz')[
        'array']

    depth_non = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality/depth_and_center-line/depth_array/patient-id-24534412.npz')[
        'array']

    normalized_cta_vessel, flow_cta = down_sample_central_mass_center_and_crop_size(vessel_cta)
    normalized_non_vessel, flow_non = down_sample_central_mass_center_and_crop_size(vessel_non)

    print(dice_score_two_class(normalized_cta_vessel, normalized_non_vessel))

    rescaled_cta_denoise = np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/rescaled_ct-denoise/patient-id-24534412.npz')['array']
    rescaled_ct_denoise = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality/rescaled_ct-denoise/patient-id-24534412.npz')['array']
    # direct ncc tensor(-0.6427)

    rescaled_cta_denoise = rescaled_cta_denoise + depth_cta / np.max(depth_cta) * 2
    rescaled_ct_denoise = rescaled_ct_denoise + depth_non / np.max(depth_non) * 2

    normalized_cta = transformation_on_array(rescaled_cta_denoise, flow_cta)
    normalized_non = transformation_on_array(rescaled_ct_denoise, flow_non)

    normalized_cta = torch.FloatTensor(normalized_cta).unsqueeze(0).unsqueeze(0)
    normalized_non = torch.FloatTensor(normalized_non).unsqueeze(0).unsqueeze(0)

    ncc_loss_normalized = weighted_ncc_loss(normalized_non, normalized_cta, show=True, to_cuda=False, return_cc_map=True)

    ncc_loss_normalized = ncc_loss_normalized.detach().cpu().numpy()[0, 0]

    sorted_value_normalized = Functions.get_sorted_values_from_given_region(ncc_loss_normalized, np.clip(normalized_non_vessel + normalized_cta_vessel, 0, 1))
    print(np.median(sorted_value_normalized))

    rescaled_cta_denoise = torch.FloatTensor(rescaled_cta_denoise).unsqueeze(0).unsqueeze(0)
    rescaled_ct_denoise = torch.FloatTensor(rescaled_ct_denoise).unsqueeze(0).unsqueeze(0)

    ncc_loss_original = weighted_ncc_loss(rescaled_ct_denoise, rescaled_cta_denoise, show=True, to_cuda=False, return_cc_map=True)
    ncc_loss_original = ncc_loss_original.detach().cpu().numpy()[0, 0]

    sorted_value_original = Functions.get_sorted_values_from_given_region(ncc_loss_original, np.clip(vessel_cta + vessel_non, 0, 1))
    print(np.median(sorted_value_original))
