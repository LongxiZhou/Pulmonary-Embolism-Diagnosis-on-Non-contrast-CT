import numpy as np
import torch
import os

"""
https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Vector_Calculus_(Corral)/04%3A_Line_and_Surface_Integrals/4.06%3A_Gradient_Divergence_Curl_and_Laplacian
"""


def get_gradient_tensor(input_tensor):
    """

    :param input_tensor: F, a multi-dimension tensor
    :return: Fx, Fy, Fz, ...
    """
    return torch.gradient(input_tensor)


def get_gradient_array(input_array):
    """

    :param input_array: F, a multi-dimension tensor
    :return: Fx, Fy, Fz, ...
    """
    return np.gradient(input_array)


def get_divergence_tensor(input_tensor_list):
    """

    apply inner product of Laplace operator

    :param input_tensor_list: [F1, F2, F3, ...], the vector field is F1 i + F2 j + F3 k + ...
    :return: divergence
    """
    num_dims = len(input_tensor_list)
    assert num_dims > 0
    divergence_tensor = torch.gradient(input_tensor_list[0], dim=0)[0]

    for dim in range(1, num_dims):
        divergence_tensor = divergence_tensor + torch.gradient(input_tensor_list[dim], dim=dim)[0]
    return divergence_tensor


def get_curl_tensor(input_tensor_list):
    """

    apply cross product of Laplace operator

    :param input_tensor_list: [P, Q, R], the vector field is P i + Q j + R k
    :return: (C1, C2, C3), the vector field is (Ry - Qz) i + (Pz - Rx) j + (Qx - Py) k
    """
    assert len(input_tensor_list) == 3

    py, pz = torch.gradient(input_tensor_list[0], dim=(1, 2))
    qx, qz = torch.gradient(input_tensor_list[1], dim=(0, 2))
    rx, ry = torch.gradient(input_tensor_list[2], dim=(0, 1))

    return ry - qz, pz - rx, qx - py


def get_divergence_array(input_array_list):
    """

    apply inner product of Laplace operator

    :param input_array_list: [F1, F2, F3, ...], the vector field is F1 i + F2 j + F3 k + ...
    :return: divergence
    """
    num_dims = len(input_array_list)
    assert num_dims > 0
    divergence_array = np.gradient(input_array_list[0], axis=0)

    for dim in range(1, num_dims):
        divergence_array = divergence_array + np.gradient(input_array_list[dim], axis=dim)
    return divergence_array


def get_curl_array(input_array_list):
    """

    apply cross product of Laplace operator

    :param input_array_list: [P, Q, R], the vector field is P i + Q j + R k
    :return: (C1, C2, C3), the vector field is (Ry - Qz) i + (Pz - Rx) j + (Qx - Py) k
    """
    assert len(input_array_list) == 3

    py, pz = np.gradient(input_array_list[0], axis=(1, 2))
    qx, qz = np.gradient(input_array_list[1], axis=(0, 2))
    rx, ry = np.gradient(input_array_list[2], axis=(0, 1))

    return ry - qz, pz - rx, qx - py


def inner_product_vector_field(input_tensor_list_a, input_tensor_list_b):
    """

    :param input_tensor_list_a: (u1, u2, u3)
    :param input_tensor_list_b: (v1, v2, v3)
    :return: u1 * v1 + u2 * v2 + u3 * v3
    """
    assert len(input_tensor_list_a) == len(input_tensor_list_b)
    num_dims = len(input_tensor_list_a)
    assert num_dims > 0

    inner_product = input_tensor_list_a[0] * input_tensor_list_b[0]
    for dim in range(1, num_dims):
        inner_product = inner_product + input_tensor_list_a[dim] * input_tensor_list_b[dim]

    return inner_product


def cross_product_vector_field(input_tensor_list_a, input_tensor_list_b):
    """

    :param input_tensor_list_a: (u1, u2, u3)
    :param input_tensor_list_b: (v1, v2, v3)
    :return: u2v3 - u3v2, - u1v3 + u3v1, u1v2 - u2v1
    """

    assert len(input_tensor_list_a) == len(input_tensor_list_b)
    num_dims = len(input_tensor_list_a)
    assert num_dims == 3

    u1, u2, u3 = input_tensor_list_a
    v1, v2, v3 = input_tensor_list_b

    return u2 * v3 - u3 * v2, - u1 * v3 + u3 * v1, u1 * v2 - u2 * v1


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions
    registration_flow = Functions.pickle_load_object(
        '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/optimal/'
        'registration_flow/patient-id-20567360.pickle')
    print(registration_flow)
    exit()
    test_vector_field = [torch.normal(0, 1, size=(100, 100, 100)).cuda(),
                         torch.normal(0, 1, size=(100, 100, 100)).cuda(),
                         torch.normal(0, 1, size=(100, 100, 100)).cuda()]

    rotate = get_curl_tensor(test_vector_field)

    di = get_divergence_tensor(rotate)

    print(torch.sum(torch.abs(di)) / 100 / 100 / 100)

    exit()
    test_array = np.load('/data_disk/artery_vein_project/new_data/CTA/rescaled_ct-denoise/AL00002.npz')['array']
    test_tensor = torch.FloatTensor(test_array).cuda()
    f = get_gradient_tensor(test_tensor)

    rotate = get_curl_tensor(f)
    print(torch.sum(torch.abs(rotate[0])), torch.sum(torch.abs(rotate[1])), torch.sum(torch.abs(rotate[2])))
    exit()
    f = [torch.normal(0, 1, size=(100, 100, 100)).cuda(),
         torch.normal(0, 1, size=(100, 100, 100)).cuda(),
         torch.normal(0, 1, size=(100, 100, 100)).cuda()]

    g = [torch.normal(0, 1, size=(100, 100, 100)).cuda(),
         torch.normal(0, 1, size=(100, 100, 100)).cuda(),
         torch.normal(0, 1, size=(100, 100, 100)).cuda()]

    a = get_divergence_tensor(cross_product_vector_field(f, g))
    b = inner_product_vector_field(get_curl_tensor(f), g) - inner_product_vector_field(f, get_curl_tensor(g))

    print(torch.sum(torch.abs(a - b)))

    exit()
