import format_convert.spatial_normalize as spatial_normalize
from registration_pulmonary.loss_functions.flow_based_loss import get_jacobi_high_precision, flow_tension_loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


def down_sample_up_sample_flow(flow_on_cpu, bottle_neck_shape=(512, 512, 512)):
    """

    :param flow_on_cpu: numpy float32 array in shape (1, 3, 256, 256, 256)
    :param bottle_neck_shape:
    :return: numpy float32 array in shape (1, 3, 256, 256, 256)
    """
    shape_flow = (1, 3, 256, 256, 256)
    smoothed_flow = np.zeros(shape_flow, 'float32')
    assert np.shape(flow_on_cpu) == shape_flow
    down_sampled_flow = np.zeros((1, 3, bottle_neck_shape[0], bottle_neck_shape[1], bottle_neck_shape[2]), 'float32')

    for i in range(3):
        down_sampled_flow[0, i] = spatial_normalize.rescale_to_new_shape(flow_on_cpu[0, i], bottle_neck_shape)

    for i in range(3):
        smoothed_flow[0, i] = spatial_normalize.rescale_to_new_shape(down_sampled_flow[0, i], shape_flow[2:])

    return smoothed_flow


def blur_flow_with_convolution_kernel(flow_on_cpu, blur_kernel_radius=10, blur_parameter=1., blur_type='half_decay'):
    """
    will do deep copy

    :param blur_type:
    :param flow_on_cpu: numpy float32 with shape (1, 3, 256, 256, 256)
    :param blur_kernel_radius:
    :param blur_parameter:
    :return: flow_on_cpu blurred, same data type and shape with input
    """
    if blur_parameter <= 0:
        return flow_on_cpu

    flow_padded = linear_pad_flow(flow_on_cpu, pad=blur_kernel_radius)

    convolution_layer = ConvolutionBlur(radius=blur_kernel_radius, blur_parameter=blur_parameter,
                                        blur_type=blur_type).cuda()

    flow_blur = apply_blur_convolution(flow_padded[0], convolution_layer=convolution_layer)

    flow_blur = np.stack((flow_blur, ), axis=0)  # unsqueeze the first channel

    return flow_blur


def linear_extend_pad(array_in_3d, pad):
    assert pad >= 0 and len(np.shape(array_in_3d)) == 3
    # padding will do deep copy of the array
    # support both float and int data type
    # padding value is fit according to values close to the boundary
    return np.pad(array_in_3d, pad, mode='linear_ramp')


def linear_pad_flow(flow_on_cpu, pad):
    """
    will do deep copy
    padding value is fit according to the boundary values

    :param flow_on_cpu: numpy float32 with shape (1, 3, 256, 256, 256)
    :param pad: int >= 0
    :return:  numpy float32 with shape (1, 3, 256 + 2 * pad, 256 + 2 * pad, 256 + 2 * pad)
    """
    flow_padded_x = linear_extend_pad(flow_on_cpu[0, 0], pad=pad)
    flow_padded_y = linear_extend_pad(flow_on_cpu[0, 1], pad=pad)
    flow_padded_z = linear_extend_pad(flow_on_cpu[0, 2], pad=pad)

    flow_padded = np.stack((flow_padded_x, flow_padded_y, flow_padded_z), axis=0)
    flow_padded = np.stack((flow_padded, ), axis=0)  # unsqueeze the first channel

    return flow_padded


def construct_psf_half_decay_3d(half_decay=1., radius=5):
    """
    get blur kernel for convolution
    :param half_decay: half decay of the weight
    :param radius:
    :return: numpy array in shape [radius * 2 + 1, radius * 2 + 1, radius * 2 + 1], sum up is 1
    """
    length_psf = int(radius * 2 + 1)
    center_loc = [radius, radius, radius]
    psf_func = np.zeros([length_psf, length_psf, length_psf], 'float32')
    for x in range(length_psf):
        for y in range(length_psf):
            for z in range(length_psf):
                distance_to_center = np.sqrt(
                    np.square(x - center_loc[0]) + np.square(y - center_loc[1]) + np.square(z - center_loc[2]))
                psf_func[x, y, z] = \
                    1 / (2 ** (distance_to_center / half_decay))
    return psf_func / np.sum(psf_func)  # set sum to 1


def construct_psf_gaussian_3d(std=4., radius=5):
    """
    get blur kernel for convolution
    :param std: std for the gaussian blur
    :param radius:
    :return: numpy array in shape [radius * 2 + 1, radius * 2 + 1, radius * 2 + 1], sum up is 1
    """
    length_psf = int(radius * 2 + 1)
    center_loc = [radius, radius, radius]
    psf_func = np.zeros([length_psf, length_psf, length_psf], 'float32')
    for x in range(length_psf):
        for y in range(length_psf):
            for z in range(length_psf):
                distance_square_to_center = \
                    np.square(x - center_loc[0]) + np.square(y - center_loc[1]) + np.square(z - center_loc[2])
                psf_func[x, y, z] = np.exp(- distance_square_to_center / std / 2)
    return psf_func / np.sum(psf_func)  # set sum to 1


class ConvolutionBlur(nn.Module):
    def __init__(self, radius=5, blur_parameter=2., blur_type='half_decay'):
        assert blur_type in ['half_decay', 'gaussian']
        assert blur_parameter > 0 and radius > 0 and type(radius) is int
        super(ConvolutionBlur, self).__init__()
        super().__init__()
        if blur_type is 'half_decay':
            kernel = construct_psf_half_decay_3d(half_decay=blur_parameter, radius=radius)
        elif blur_type is 'gaussian':
            kernel = construct_psf_gaussian_3d(std=blur_parameter, radius=radius)
        else:
            raise ValueError("blur type undefined:", blur_type)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, image_padded):
        image_blurred = func.conv3d(image_padded, self.weight, padding=0)
        return image_blurred


def apply_blur_convolution(float_numpy_array, convolution_layer=None,
                           radius=5, blur_parameter=2, blur_type='half_decay', return_same_shape=False):
    """

    :param return_same_shape:
    :param convolution_layer: on GPU
    :param float_numpy_array: in shape [x, y, z] or [batch, x, y, z]
    :param radius:
    :param blur_parameter:
    :param blur_type:
    :return: blurred array
    """

    if return_same_shape:
        float_numpy_array = linear_extend_pad(float_numpy_array, pad=radius)

    if convolution_layer is None:
        convolution_layer = ConvolutionBlur(radius=radius, blur_parameter=blur_parameter, blur_type=blur_type).cuda()

    shape = np.shape(float_numpy_array)
    if len(shape) == 3:
        array = torch.from_numpy(float_numpy_array).unsqueeze(0).unsqueeze(0)
    elif len(shape) == 4:
        array = torch.from_numpy(float_numpy_array).unsqueeze(1)
    else:
        raise ValueError("shape for input array have to with dimension 3 or 4")

    # now the array in shape [batch_size, 1, x, y, z]
    blurry_array = convolution_layer(array.cuda())
    blurry_array = blurry_array.to('cpu')
    blurry_array = blurry_array.data.numpy()

    if len(shape) == 3:
        return blurry_array[0, 0, :, :, :]  # [x, y, z]
    else:
        return blurry_array[:, 0, :, :, :]  # [batch_size, x, y, z]


def up_sample_256_flow_to_512(flow_256):
    assert np.shape(flow_256) == (1, 3, 256, 256, 256)
    flow_512 = np.zeros((1, 3, 512, 512, 512), 'float32')
    for axis in range(3):
        flow_512[0, axis] = spatial_normalize.rescale_to_new_shape(
            flow_256[0, axis], (512, 512, 512), change_format=False)
    return flow_512


def down_sample_512_flow_to_256(flow_512):
    assert np.shape(flow_512) == (1, 3, 512, 512, 512)
    flow_256 = np.zeros((1, 3, 256, 256, 256), 'float32')
    for axis in range(3):
        flow_256[0, axis] = spatial_normalize.rescale_to_new_shape(
            flow_512[0, axis], (256, 256, 256), change_format=False)
    return flow_256


def up_sample_256_flow_to_512_v2(flow_256):
    assert np.shape(flow_256) == (1, 3, 256, 256, 256)

    def resize_img(array, factor=2):
        from torch.nn.functional import interpolate
        mode = "trilinear"
        tensor = torch.FloatTensor(array)
        tensor = interpolate(tensor, scale_factor=factor, mode=mode)
        return tensor.cpu().numpy()

    flow_512 = resize_img(flow_256)
    assert np.shape(flow_256) == (1, 3, 512, 512, 512)
    return flow_512


def show_example_down_sample_up_sample_smooth(scan_name='patient-id-135'):
    from pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database import \
        load_flow_cta_to_non_contrast
    import Tool_Functions.Functions as Functions
    import torch

    flow_combined, performance_dict = load_flow_cta_to_non_contrast(scan_name)
    # flow_combined: [normalization_flow_cta, registration_flow, normalization_flow_non_contrast]

    print("shape_flow_smoothed:", np.shape(flow_combined[1]))

    flow_original = torch.FloatTensor(flow_combined[1])

    jacobi_tensor_original = get_jacobi_high_precision(flow_original)
    jacobi_original = jacobi_tensor_original.numpy()
    tension_loss_original = flow_tension_loss(jacobi_tensor_original).numpy()
    print("min jacobi", np.min(jacobi_original), "   flow tension loss", tension_loss_original)

    flow_smoothed = down_sample_up_sample_flow(flow_combined[1])
    print("shape_flow_smoothed:", np.shape(flow_smoothed))

    jacobi_tensor_smoothed = get_jacobi_high_precision(torch.FloatTensor(flow_smoothed))
    jacobi_smoothed = jacobi_tensor_smoothed.numpy()
    tension_loss_smoothed = flow_tension_loss(jacobi_tensor_smoothed).numpy()
    print("min jacobi", np.min(jacobi_smoothed), "   flow tension loss", tension_loss_smoothed)
    array_lower_than_zero = np.array(jacobi_original < 0, 'int16')
    print(np.shape(array_lower_than_zero))
    print("ratio lower than zero original:", np.sum(array_lower_than_zero) / (np.shape(array_lower_than_zero)[1] ** 3))
    Functions.show_multiple_images(original=jacobi_original[0, :, :, 128], smoothed=jacobi_smoothed[0, :, :, 128])


def show_example_blur_smooth(scan_name='patient-id-135', up_sample_smooth=False):
    from pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database import \
        load_flow_cta_to_non_contrast
    import Tool_Functions.Functions as Functions

    flow_combined, performance_dict = load_flow_cta_to_non_contrast(scan_name)
    # flow_combined: [normalization_flow_cta, registration_flow, normalization_flow_non_contrast]

    print("shape_flow_original:", np.shape(flow_combined[1]))

    flow_original = flow_combined[1]

    print("\njacobi original")
    jacobi_original = show_jacobi_of_flow(flow_original)

    flow_smoothed = blur_flow_with_convolution_kernel(
        flow_combined[1], blur_kernel_radius=20, blur_parameter=2, blur_type='half_decay')
    print("shape_flow_smoothed:", np.shape(flow_smoothed))

    if up_sample_smooth:
        flow_smoothed = up_sample_256_flow_to_512(flow_smoothed)
        # flow_smoothed = down_sample_up_sample_flow(flow_smoothed)

        flow_smoothed_256 = down_sample_512_flow_to_256(flow_smoothed)
        print("\njacobi smoothed 256")
        jacobi_smoothed_256 = show_jacobi_of_flow(flow_smoothed_256)
        Functions.show_multiple_images(
            original=jacobi_original[0, :, :, 128], smoothed=jacobi_smoothed_256[0, :, :, 128])

    print("\njacobi smoothed")
    jacobi_smoothed = show_jacobi_of_flow(flow_smoothed)

    if up_sample_smooth:
        Functions.show_multiple_images(original=jacobi_original[0, :, :, 128], smoothed=jacobi_smoothed[0, :, :, 256])
    else:
        Functions.show_multiple_images(original=jacobi_original[0, :, :, 128], smoothed=jacobi_smoothed[0, :, :, 128])


def smooth_256_then_up_sample_to_512(
        flow_numpy_array, blur_kernel_radius=20, blur_parameter=2, blur_type='half_decay', show_jacobi=False):
    assert np.shape(flow_numpy_array) == (1, 3, 256, 256, 256)
    flow_blurred_256 = blur_flow_with_convolution_kernel(
        flow_numpy_array, blur_kernel_radius=blur_kernel_radius, blur_parameter=blur_parameter, blur_type=blur_type)

    flow_blurred_512 = up_sample_256_flow_to_512(flow_blurred_256)
    if show_jacobi:
        print("\noriginal flow 256")
        show_jacobi_of_flow(flow_numpy_array)
        print("\nsmoothed flow 256")
        show_jacobi_of_flow(flow_blurred_256)
        print("\nsmoothed then up-sampled flow 512")
        show_jacobi_of_flow(flow_blurred_512)
    return flow_blurred_512


def show_jacobi_of_flow(flow_numpy_array):
    flow_tensor = torch.FloatTensor(flow_numpy_array)
    jacobi_tensor = get_jacobi_high_precision(flow_tensor)
    jacobi_array = jacobi_tensor.numpy()
    tension_loss_original = flow_tension_loss(jacobi_tensor).numpy()
    array_lower_than_zero = np.array(jacobi_array < 0, 'int16')
    print("flow shape:", np.shape(array_lower_than_zero))
    print("ratio lower than zero:", np.sum(array_lower_than_zero) / (np.shape(array_lower_than_zero)[1] ** 3))
    print("min jacobi", np.min(jacobi_array), "   flow tension loss", tension_loss_original)
    return jacobi_array


if __name__ == '__main__':

    show_example_blur_smooth(up_sample_smooth=True)
    exit()
