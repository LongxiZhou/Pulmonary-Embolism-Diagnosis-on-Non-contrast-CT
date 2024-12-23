
import torch
from collaborators_package.chest_register.registration.models.register import VxmDense as Register_1
from torch.nn.functional import interpolate
import numpy as np
import format_convert.basic_transformations as basic_transform


# Training settings
def down_sample_img(tensor, factor):
    mode = "trilinear"
    tensor = interpolate(tensor, scale_factor=factor, mode=mode)
    return tensor


def down_sample_seg(tensor, factor):
    mode = "trilinear"
    tensor = interpolate(tensor, scale_factor=factor, mode=mode)
    tensor[tensor > 0.1] = 1
    return tensor


def preprocess(array, device=torch.device("cuda")):
    array = array[np.newaxis, np.newaxis, :]
    array = torch.FloatTensor(array).to(device)
    return array


def register_with_given_flow(numpy_array, flow_on_cpu, device=torch.device("cuda")):
    from collaborators_package.chest_register.registration.models import layers
    # numpy array in shape like (256, 256, 256), flow in shape like (1, 3, 256, 256, 256)
    tensor_moving = preprocess(numpy_array, device)

    flow_on_gpu = torch.FloatTensor(flow_on_cpu).to(device)

    transform_with_flow = layers.SpatialTransformer(np.shape(numpy_array))
    transform_with_flow = transform_with_flow.to(device)

    registered_tensor = transform_with_flow(tensor_moving, flow_on_gpu)

    return registered_tensor.cpu().detach().numpy()[0, 0]


def register_with_flow_combine(numpy_array, registration_flow_combine):
    """

    :param registration_flow_combine: [normalization_flow_moving, registration_flow, normalization_flow_fix]
    :param numpy_array: numpy array in float32
    :return: array_registered_to_non_contrast, in float32
    """
    normalization_flow_moving, registration_flow, normalization_flow_fix = registration_flow_combine

    normalized_array = basic_transform.transformation_on_array(numpy_array, normalization_flow_moving, reverse=False)

    # registered to normalized non-contrast, normalized array in shape (256, 256, 256)
    # flow in shape (1, 3, 256, 256, 256)
    registered_array = register_with_given_flow(normalized_array, registration_flow)

    # undo normalization
    array_registered_to_non_contrast = basic_transform.transformation_on_array(
        registered_array, normalization_flow_fix, reverse=True)

    return array_registered_to_non_contrast


def smooth_guide_mask(rescaled_ct, guide_mask):
    import analysis.get_surface_rim_adjacent_mean as get_surface
    import Tool_Functions.Functions as Functions
    if rescaled_ct is not None:
        guide_mask = guide_mask * np.array(rescaled_ct > Functions.change_to_rescaled(-200), 'float32')
        # smooth
    guide_mask = guide_mask + get_surface.get_surface(guide_mask, outer=True, strict=False)
    guide_mask = guide_mask - get_surface.get_surface(guide_mask, outer=False, strict=False)

    return guide_mask


def register(moving_img, fixed_img, moving_seg, fixed_seg, two_stage=True, down_sample=True,
             return_flow=False, device=torch.device("cuda")):
    """

    :param moving_img: numpy [512, 512, 512] or [256, 256, 256]
    :param fixed_img: numpy [512, 512, 512] or [256, 256, 256]
    :param moving_seg: blood vessel
    :param fixed_seg: blood vessel
    :param return_flow: return the transform flow on CPU, numpy [1, 3, 256, 256, 256]
    :param device:
    :param down_sample:
    :return: registered_img, registered_seg, fixed_img, fixed_seg, register_flow (optional)
    """

    if moving_seg is None or fixed_seg is None:
        moving_seg = 0 * moving_img
        fixed_seg = 0 * fixed_img

    if down_sample:
        assert np.shape(moving_img) == (512, 512, 512)
        moving_img = down_sample_img(preprocess(moving_img, device), factor=1 / 2)
        moving_seg = down_sample_seg(preprocess(moving_seg, device), factor=1 / 2)
        fixed_img = down_sample_img(preprocess(fixed_img, device), factor=1 / 2)
        fixed_seg = down_sample_seg(preprocess(fixed_seg, device), factor=1 / 2)
    else:
        assert np.shape(moving_img) == (256, 256, 256)
        moving_img = preprocess(moving_img, device)
        moving_seg = preprocess(moving_seg, device)
        fixed_img = preprocess(fixed_img, device)
        fixed_seg = preprocess(fixed_seg, device)

    # print("===> Building model_0")
    scale_factor = 1  # down sample the image by scale factor to get flow, then upsample the flow
    # higher the factor, the more coarse the flow but wider ROI.
    vol_size = [256, 256, 256]
    nf_enc = [16, 32, 64, 128]
    nf_dec = [128, 128, 64, 64, 32, 32, 16]
    model_0 = Register_1(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    # print("===> Loading datasets")
    model_0.load_state_dict(
        torch.load("/data_disk/Altolia_share/register/model_epoch_1260.pth"))
    model_0 = model_0.to('cuda:0')
    registered_img_0, registered_seg_0, pos_flow_0 = model_0(moving_img, fixed_img, moving_seg, fixed_seg)

    if not two_stage:
        if return_flow:
            return registered_img_0.cpu().detach().numpy()[0, 0], \
                   registered_seg_0.cpu().detach().numpy()[0, 0], \
                   fixed_img.cpu().detach().numpy()[0, 0], \
                   fixed_seg.cpu().detach().numpy()[0, 0], \
                   pos_flow_0.cpu().detach().numpy()

        return registered_img_0.cpu().detach().numpy()[0, 0], \
               registered_seg_0.cpu().detach().numpy()[0, 0], \
               fixed_img.cpu().detach().numpy()[0, 0], \
               fixed_seg.cpu().detach().numpy()[0, 0]

    # print("===> Building model_1")
    scale_factor = 1
    vol_size = [256, 256, 256]
    nf_enc = [16, 32, 64, 128]
    nf_dec = [128, 128, 64, 64, 32, 32, 16]
    model_1 = Register_1(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    # print("===> Loading datasets")
    model_1.load_state_dict(
        torch.load("/data_disk/Altolia_share/register/model_epoch_26.pth"))
    model_1 = model_1.to('cuda:0')
    (registered_img_1, registered_seg_1, pos_flow_1) = \
        model_1(registered_img_0, fixed_img, registered_seg_0, fixed_seg)
    if return_flow:
        return registered_img_1.cpu().detach().numpy()[0, 0], \
               registered_seg_1.cpu().detach().numpy()[0, 0], \
               fixed_img.cpu().detach().numpy()[0, 0], \
               fixed_seg.cpu().detach().numpy()[0, 0], \
               pos_flow_0.cpu().detach().numpy(), pos_flow_1.cpu().detach().numpy()

    return registered_img_1.cpu().detach().numpy()[0, 0], \
           registered_seg_1.cpu().detach().numpy()[0, 0], \
           fixed_img.cpu().detach().numpy()[0, 0], \
           fixed_seg.cpu().detach().numpy()[0, 0]


def normalization(binary_mask):
    """

    change shape from 512 to  256,
    set mask mass center to [128, 128, 128]

    :param binary_mask:
    :return: normalized_mask, transformation_flow
    """

    return basic_transform.down_sample_central_mass_center_and_crop_size(binary_mask, crop=False)


if __name__ == '__main__':
    def dice(array_1, array_2):
        inter = np.sum(array_1 * array_2)
        norm = np.sum(array_1 * array_1) + np.sum(array_2 * array_2)
        return 2 * inter / norm

    blood_exhale = np.load('/data_disk/lung_altas/inhale_exhale_pair_one_patient/semantics/blood_mask/S50.npz')['array']
    normalized_blood_exhale, flow_exhale = normalization(blood_exhale)

    blood_inhale = np.load('/data_disk/lung_altas/inhale_exhale_pair_one_patient/semantics/blood_mask/S30.npz')['array']
    normalized_blood_inhale, flow_inhale = normalization(blood_inhale)

    print("dice on original:", dice(blood_inhale, blood_exhale))
    print("dice on normalized:", dice(normalized_blood_exhale, normalized_blood_inhale))

    ct_exhale = np.load('/data_disk/lung_altas/inhale_exhale_pair_one_patient/rescaled_ct-denoise/S50.npz')['array']
    normalized_ct_exhale = basic_transform.transformation_on_array(ct_exhale, flow_exhale)

    ct_inhale = np.load('/data_disk/lung_altas/inhale_exhale_pair_one_patient/rescaled_ct-denoise/S30.npz')['array']
    normalized_ct_inhale = basic_transform.transformation_on_array(ct_inhale, flow_inhale)

    # register exhale to inhale
    registered_img, registered_seg, fixed_img, fixed_seg, register_flow = register(
        normalized_ct_exhale, normalized_ct_inhale, normalized_blood_exhale, normalized_blood_inhale,
        two_stage=False, down_sample=False, return_flow=True)

    print(dice(fixed_seg, registered_seg))

    from analysis.differential_and_jacobi import calculate_jacobi_registration_flow
    import Tool_Functions.Functions as Functions

    flow_d = torch.FloatTensor(register_flow)

    extend_tensor = calculate_jacobi_registration_flow(flow_d[0, 0], flow_d[0, 1], flow_d[0, 2])

    extend_tensor = extend_tensor.detach().numpy()

    print(np.min(extend_tensor), np.max(extend_tensor))
    extend_tensor = np.clip(extend_tensor, 1, 5) - 1
    extend_tensor = extend_tensor / np.max(extend_tensor)

    print(np.shape(extend_tensor))

    for z in range(100, 300, 10):
        print(z)
        Functions.merge_image_with_mask(normalized_ct_exhale[:, :, z], extend_tensor[:, :, z])
        Functions.merge_image_with_mask(normalized_ct_inhale[:, :, z], extend_tensor[:, :, z])
