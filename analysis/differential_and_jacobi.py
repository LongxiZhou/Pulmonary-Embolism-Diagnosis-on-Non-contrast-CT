import numpy as np
import torch


def calculate_jacobi_registration_flow(fx, fy, fz, precision=4):
    """

    registration flow = fx * i + fy *j + fz * k
    registration_flow records how each voxel move during registration

    :param fx: tensor in shape [x, y, z]
    :param fy: tensor in shape [x, y, z]
    :param fz: tensor in shape [x, y, z]
    :param precision: in 1, 2, 4, the error is O(h^precision), h is the voxel length
    :return: tensor in shape [x, y, z], each voxel, is the number of times of volume change after registration
    """
    fx_x = differential_on_axis_tensor(fx, axis=0, precision=precision)
    fx_y = differential_on_axis_tensor(fx, axis=1, precision=precision)
    fx_z = differential_on_axis_tensor(fx, axis=2, precision=precision)

    fy_x = differential_on_axis_tensor(fy, axis=0, precision=precision)
    fy_y = differential_on_axis_tensor(fy, axis=1, precision=precision)
    fy_z = differential_on_axis_tensor(fy, axis=2, precision=precision)

    fz_x = differential_on_axis_tensor(fz, axis=0, precision=precision)
    fz_y = differential_on_axis_tensor(fz, axis=1, precision=precision)
    fz_z = differential_on_axis_tensor(fz, axis=2, precision=precision)

    jacobi_determinant = get_33_determinant(fx_x + 1, fx_y, fx_z, fy_x, fy_y + 1, fy_z, fz_x, fz_y, fz_z + 1)

    return jacobi_determinant


def get_33_determinant(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    determinant = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    return determinant


def differential_on_axis_tensor(tensor, axis, precision=4):
    """

    :param tensor: torch float tensor in any shape.
    :param axis: int
    :param precision: in 1, 2, 4, the error is O(h^precision), h is the voxel length
    :return: same shape with input (values in the boundary has lower precision)
    """
    shape_tensor = tensor.shape
    shape_at_dim = shape_tensor[axis]
    assert shape_at_dim > precision

    if precision == 1:
        new_tensor = torch.diff(tensor, n=1, dim=axis)
        pad_last = new_tensor[(slice(None), ) * axis + (slice(shape_tensor[axis] - 2, shape_tensor[axis] - 1), )]
        new_tensor = torch.cat((new_tensor, pad_last), dim=axis)
        return new_tensor
    elif precision == 2:
        tensor_a = tensor[(slice(None), ) * axis + (slice(2, shape_tensor[axis]), )]
        tensor_b = tensor[(slice(None),) * axis + (slice(0, shape_tensor[axis] - 2),)]

        new_tensor = (tensor_a - tensor_b) / 2

        slice_0 = tensor[(slice(None),) * axis + (slice(0, 1),)]
        slice_1 = tensor[(slice(None),) * axis + (slice(1, 2),)]
        pad_first = slice_1 - slice_0  # precision 1
        pad_last = new_tensor[(slice(None),) * axis + (slice(shape_tensor[axis] - 3, shape_tensor[axis] - 2),)]
        new_tensor = torch.cat((pad_first, new_tensor, pad_last), dim=axis)
        return new_tensor
    else:
        assert precision == 4
        tensor_a = tensor[(slice(None),) * axis + (slice(4, shape_tensor[axis]),)]
        tensor_b = tensor[(slice(None),) * axis + (slice(3, shape_tensor[axis] - 1),)]
        tensor_c = tensor[(slice(None),) * axis + (slice(1, shape_tensor[axis] - 3),)]
        tensor_d = tensor[(slice(None),) * axis + (slice(0, shape_tensor[axis] - 4),)]

        new_tensor = -tensor_a + 8 * tensor_b - 8 * tensor_c + tensor_d
        new_tensor = new_tensor / 12

        slice_0 = tensor[(slice(None),) * axis + (slice(0, 1),)]
        slice_1 = tensor[(slice(None),) * axis + (slice(1, 2),)]
        slice_2 = tensor[(slice(None),) * axis + (slice(2, 3),)]

        pad_first = slice_1 - slice_0  # precision 1
        pad_second = (slice_2 - slice_0) / 2  # precision 2

        slice_last = tensor[(slice(None),) * axis + (slice(shape_tensor[axis] - 1, shape_tensor[axis]),)]
        slice_last_two = tensor[(slice(None),) * axis + (slice(shape_tensor[axis] - 2, shape_tensor[axis] - 1),)]
        pad_last_two = slice_last_two - slice_last  # precision 1
        pad_last = pad_last_two

        new_tensor = torch.cat((pad_first, pad_second, new_tensor, pad_last_two, pad_last), dim=axis)
        return new_tensor


if __name__ == '__main__':

    from visualization.visiualize_2d.image_visualization import show_2d_function
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    def sin_cos(loc):
        x, y = loc
        return np.sin(x) * np.cos(y)

    def sin_cos_dif(loc):
        x, y = loc
        return np.cos(x) * np.cos(y)

    test_array = show_2d_function(sin_cos, (0, 8), (0, 8), resolution=(8, 8), show=False)
    real_diff_axis_1 = show_2d_function(sin_cos_dif, (0, 8), (0, 8), resolution=(8, 8), show=False)
    print(np.shape(test_array))
    print(np.shape(real_diff_axis_1))

    print(test_array)
    print(real_diff_axis_1)
    test_tensor = torch.FloatTensor(test_array)
    print(differential_on_axis_tensor(test_tensor, 1, precision=4))
    print(differential_on_axis_tensor(test_tensor, 1, precision=2))
    print(differential_on_axis_tensor(test_tensor, 1, precision=1))
