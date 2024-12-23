"""
loss functions calculated based on registration flow
"""

import torch
import analysis.differential_and_jacobi as get_jacobi


def gradient_loss_l2(registration_flow):
    """
    penalty on sudden change on
    :param registration_flow:
    :return:
    """

    dy = registration_flow[:, :, 1:, :, :] - registration_flow[:, :, :-1, :, :]
    dx = registration_flow[:, :, :, 1:, :] - registration_flow[:, :, :, :-1, :]
    dz = registration_flow[:, :, :, :, 1:] - registration_flow[:, :, :, :, :-1]

    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3


def get_jacobi_low_precision(registration_flow):
    """

    :param registration_flow: torch float tensor in shape [B, 3, x, y, z]
    :return: [B, x - 1, y - 1, z - 1]
    """

    f_x = (registration_flow[:, :, 1:, :-1, :-1] - registration_flow[:, :, :-1, :-1, :-1])
    f_y = (registration_flow[:, :, :-1, 1:, :-1] - registration_flow[:, :, :-1, :-1, :-1])
    f_z = (registration_flow[:, :, :-1, :-1, 1:] - registration_flow[:, :, :-1, :-1, :-1])

    f1 = (f_x[:, 0] + 1) * ((f_y[:, 1] + 1) * (f_z[:, 2] + 1) - f_z[:, 1] * f_y[:, 2])
    f2 = (f_x[:, 1]) * (f_y[:, 0] * (f_z[:, 2] + 1) - f_y[:, 2] * f_x[:, 0])
    f3 = (f_x[:, 2]) * (f_y[:, 0] * f_z[:, 1] - (f_y[:, 1] + 1) * f_z[:, 0])

    return f1 - f2 + f3


def get_jacobi_high_precision(registration_flow, precision=4):
    """

    :param registration_flow: torch float tensor in shape [B, 3, x, y, z]
    :param precision: the error is O(h^precision), h is the voxel length
    :return: torch float tensor in shape [B, x, y, z]
    """
    assert precision in [1, 2, 4]
    batch_size = registration_flow.shape[0]
    tensor_list = []
    for i in range(batch_size):
        jacobi_determinant_tensor = get_jacobi.calculate_jacobi_registration_flow(
            registration_flow[i, 0], registration_flow[i, 1], registration_flow[i, 2], precision=precision)

        tensor_list.append(jacobi_determinant_tensor)

    return torch.stack(tensor_list, dim=0)


def negative_jacobi_loss(jacobi_determinant_tensor):
    return torch.sum(torch.abs(jacobi_determinant_tensor) - jacobi_determinant_tensor) / 2


def flow_tension_loss(jacobi_determinant_tensor, penalty_weight=None, base=0.01):
    """
    estimate the energy needed to form the registration field

    :param jacobi_determinant_tensor: torch float tensor in shape [B, x, y, z]
    :param penalty_weight: stiffness or relative focus for each voxel. torch float tensor in shape [B, x, y, z]
    :param base: use base to avoid zero division
    :return: estimated energy
    """

    relative_shape = torch.abs(jacobi_determinant_tensor) + base  # use base to avoid zero division

    energy_extend = torch.square(relative_shape - 1)  # dominated by extension
    energy_compress = torch.square((1 / relative_shape) - 1)  # dominated by compression

    energy_tensor = energy_extend + energy_compress

    if penalty_weight is not None:
        energy_tensor = energy_tensor * penalty_weight

    return torch.mean(energy_tensor) * (base * 20)  # set it around 0.5


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    flow_o = Functions.pickle_load_object(
        '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/optimal/'
        'registration_flow/11.17p13.pickle')[1]
    flow_o = torch.FloatTensor(flow_o)
    print(flow_o.shape)  # torch.Size([1, 3, 256, 256, 256])
    print('gradient_loss', gradient_loss_l2(flow_o))

    print("tension_loss", flow_tension_loss(get_jacobi_high_precision(flow_o)))
    exit()

    v1 = get_jacobi_high_precision(flow_o, precision=4)  # there will be ~ 5% difference in precision 1 and precision 4

    print(v1.shape)  # torch.Size([1, 256, 256, 256])
    v1 = v1[0, 1: 254, 1: 254, 1: 254]
    print("energy need v1", flow_tension_loss(v1))
    print("negative jacobi v1", negative_jacobi_loss(v1))

    v2 = get_jacobi_low_precision(flow_o)[0, 1: 254, 1: 254, 1: 254]
    print(v2.shape)  # torch.Size([255, 255, 255])
    print("energy need v2", flow_tension_loss(v2))
    print("negative jacobi v2", negative_jacobi_loss(v2))

    print(torch.mean(torch.abs(v1 - v2)))  # tensor(0.0039)
    print(torch.mean(v1))  # tensor(0.9990)
    print(torch.mean(v2))  # tensor(0.9992)
