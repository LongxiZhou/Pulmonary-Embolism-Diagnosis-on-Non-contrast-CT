"""
given a binary mask, return its rim or surface
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func


class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


class GetRimLoose(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetRimLoose, self).__init__()
        super().__init__()
        kernel = [[[[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv2d(x, self.weight, padding=1)
        return x


class GetRimStrict(nn.Module):
    # if the adjacency is strictly defined, use this to get rim
    def __init__(self):
        super(GetRimStrict, self).__init__()
        super().__init__()
        kernel = [[[[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv2d(x, self.weight, padding=1)
        return x


class GetSurfaceLoose(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetSurfaceLoose, self).__init__()
        super().__init__()
        kernel = [[[[[-1, -1, -1],
                     [-1, -1, -1],
                     [-1, -1, -1]],
                    [[-1, -1, -1],
                     [-1, 26, -1],
                     [-1, -1, -1]],
                    [[-1, -1, -1],
                     [-1, -1, -1],
                     [-1, -1, -1]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


class GetSurfaceStrict(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetSurfaceStrict, self).__init__()
        super().__init__()
        kernel = [[[[[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]],
                    [[0, -1, 0],
                     [-1, 6, -1],
                     [0, -1, 0]],
                    [[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


class GroupAdjacentValues(nn.Module):
    """
    central pixel take weight 1, the six most adjacent pixels take weight 1/6, the 12 next most adjacent pixels take
    weight 1/12 (distance sqrt(2), weight 1 / 6 / 2), the 8 pixels with distance sqrt(3) with weight 1 / 6 / 3.
    """
    def __init__(self, kernel=None, padding=None):
        super(GroupAdjacentValues, self).__init__()
        super().__init__()
        if kernel is None:
            kernel = [[[[[1/18, 1/12, 1/18],
                         [1/12, 1/6, 1/12],
                         [1/18, 1/12, 1/18]],
                        [[1/12, 1/6, 1/12],
                         [1/6, 1, 1/6],
                         [1/12, 1/6, 1/12]],
                        [[1/18, 1/12, 1/18],
                         [1/12, 1/6, 1/12],
                         [1/18, 1/12, 1/18]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = padding
        if padding is None:
            self.padding = 1

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=self.padding)
        return x


class GroupSum(nn.Module):
    def __init__(self, kernel=None, padding=None, sum_decay=1 / 27):
        super(GroupSum, self).__init__()
        super().__init__()
        if kernel is None:
            kernel = [[[[[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]],
                        [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]],
                        [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]]]]
        kernel = torch.FloatTensor(kernel) * sum_decay
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = padding
        if padding is None:
            self.padding = 1

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=self.padding)
        return x


def get_rim(binary_mask, outer=False, strict=True):
    """
    :param strict: see kernel in GetRimStrict and GetRimStrict
    :param binary_mask: with shape [batch size, x, y] or [x, y]
    :param outer: outer rim?
    :return: binary mask, same shape with input, with 1 means rim
    """
    binary_mask = np.array(binary_mask, 'float32')
    if not strict:
        convolution_layer = GetRimLoose().cuda()
    else:
        convolution_layer = GetRimStrict().cuda()
    if torch.cuda.device_count() > 1:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(binary_mask)
    if len(shape) == 2:
        array = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
    elif len(shape) == 3:
        array = torch.from_numpy(binary_mask).unsqueeze(1)
    else:
        raise DimensionError(binary_mask)

    # now the array in shape [batch_size, 1, x, y]
    rim = convolution_layer(array.cuda())
    rim = rim.to('cpu')
    rim = rim.data.numpy()

    if outer == 'both':
        rim = np.array(rim > 0.1, 'float32') - np.array(rim < -0.1, 'float32')
        if len(shape) == 2:
            return rim[0, 0, :, :]  # [x, y]
        else:
            return rim[:, 0, :, :]  # [batch_size, x, y]

    if outer:
        rim = np.array(rim < -0.1, 'float32')
    else:
        rim = np.array(rim > 0.1, 'float32')

    if len(shape) == 2:
        return rim[0, 0, :, :]  # [x, y]
    else:
        return rim[:, 0, :, :]  # [batch_size, x, y]


def get_surface(binary_mask, outer=False, strict=True, parallel=False):
    """
    :param parallel: use more than one GPU ?
    :param strict: see kernel in GetSurfaceStrict and GetSurfaceStrict
    :param binary_mask: with shape [batch size, x, y, z] or [x, y, z]
    :param outer: outer surface?
    :return: binary mask, same shape with input, with 1 means surface
    """

    if not strict:
        convolution_layer = GetSurfaceLoose().cuda()
    else:
        convolution_layer = GetSurfaceStrict().cuda()
    if torch.cuda.device_count() > 1 and parallel:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(binary_mask)
    if len(shape) == 3:
        array = torch.FloatTensor(torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0))
    elif len(shape) == 4:
        array = torch.FloatTensor(torch.from_numpy(binary_mask).unsqueeze(1))
    else:
        raise DimensionError(binary_mask)

    # now the array in shape [batch_size, 1, x, y, z]
    surface = convolution_layer(array.cuda())
    surface = surface.to('cpu')
    surface = surface.data.numpy()

    if outer == 'both':
        surface_outer = np.array(surface < -0.1, 'float32')
        surface_inner = np.array(surface > 0.1, 'float32')
        surface = surface_inner - surface_outer
        if len(shape) == 3:
            return surface[0, 0, :, :, :]  # [x, y, z]
        else:
            return surface[:, 0, :, :, :]  # [batch_size, x, y, z]

    if outer:
        surface = np.array(surface < -0.1, 'float32')
    else:
        surface = np.array(surface > 0.1, 'float32')

    if len(shape) == 3:
        return surface[0, 0, :, :, :]  # [x, y, z]
    else:
        return surface[:, 0, :, :, :]  # [batch_size, x, y, z]


def soft_cast_to_binary(probability_mask, threshold=0.25):
    """

    :param probability_mask:
    :param threshold:
    :return:
    """
    convolution_layer = GroupAdjacentValues().cuda()
    if torch.cuda.device_count() > 1:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(probability_mask)
    assert len(shape) == 3
    array = torch.from_numpy(probability_mask).unsqueeze(0).unsqueeze(0)

    # now the array in shape [1, 1, x, y, z]
    grouped_adjacent_array = convolution_layer(array.cuda())
    grouped_adjacent_array = grouped_adjacent_array.to('cpu')
    grouped_adjacent_array = grouped_adjacent_array.data.numpy()[0, 0, :, :, :]  # [x, y, z]

    max_value = np.max(grouped_adjacent_array)
    binary_mask = np.array(grouped_adjacent_array > max_value * threshold, 'float32')

    return binary_mask


def surrounding_sum(value_mask, sum_decay=1 / 27):
    """

    :param value_mask:
    :param sum_decay:
    :return:
    """
    convolution_layer = GroupSum(sum_decay=sum_decay).cuda()

    shape = np.shape(value_mask)
    assert len(shape) == 3
    array = torch.from_numpy(value_mask).unsqueeze(0).unsqueeze(0)

    # now the array in shape [1, 1, x, y, z]
    sum_adjacent_array = convolution_layer(array.cuda())
    sum_adjacent_array = sum_adjacent_array.to('cpu')
    sum_adjacent_array = sum_adjacent_array.data.numpy()[0, 0, :, :, :]  # [x, y, z]

    return sum_adjacent_array


def region_average_voxel_wise_3d(input_array, kernel=None, padding=None):
    """

    :param input_array: only support 3D array.
    :param kernel: format can be understood by torch.FloatTensor(kernel) and np.shape(kernel)
    :param padding:
    :return:
    """
    if kernel is None:
        kernel = np.ones([1, 1, 3, 3, 3], 'float32')

    kernel = kernel / np.sum(kernel)

    kernel_size = np.shape(kernel)[2:5]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1 and kernel_size[2] % 2 == 1

    assert len(np.shape(kernel)) == 5
    shape = np.shape(input_array)
    assert len(shape) == 3

    if padding is None:
        padding = (int(kernel_size[0] / 2), int(kernel_size[1] / 2), int(kernel_size[2] / 2))

    convolution_layer = GroupAdjacentValues(kernel=kernel, padding=padding).cuda()

    array = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
    # now the array in shape [1, 1, x, y, z]

    grouped_adjacent_array = convolution_layer(array.cuda())
    grouped_adjacent_array = grouped_adjacent_array.to('cpu')
    grouped_adjacent_array = grouped_adjacent_array.data.numpy()[0, 0, :, :, :]  # [x, y, z]

    assert np.shape(grouped_adjacent_array) == shape

    return grouped_adjacent_array


if __name__ == '__main__':
    test_array = np.load('/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/non-contrast/patient-id-554.npy')

    import Tool_Functions.Functions as Functions
    Functions.image_show(test_array[:, :, 256])
    Functions.set_visible_device('2')

    averaged_test_array = region_average_voxel_wise_3d(test_array, kernel=np.ones([1, 1, 7, 7, 5]))

    Functions.image_show(averaged_test_array[:, :, 256])

    exit()
