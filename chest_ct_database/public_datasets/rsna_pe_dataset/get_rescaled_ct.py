import SimpleITK as sitk
import format_convert.spatial_normalize as spatial_normalize
import lung_atlas.upsample_ct_z.predict_up_sampled as up_sample
import numpy as np
import os
import Tool_Functions.Functions as Functions
from format_convert.read_in_CT import stack_dcm_with_instance_id, get_resolution_from_dcm


def simple_stack_dcm_files(dcm_dict):
    return stack_dcm_with_instance_id(dcm_dict, show=True)


def simple_abstract_mha_file(mha_path, cast_to_binary=False):
    ar = sitk.ReadImage(mha_path)

    mask = sitk.GetArrayFromImage(ar)  # z y x

    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)

    if cast_to_binary:
        mask = np.array(mask > 0.5, 'float32')

    return mask  # (x, y, z)


def establish_rescale_chest_ct(dcm_dict, checkpoint_path_upsample=None, show=True, batch_size=2,
                               return_original_resolution=False, original_resolution=None, clip_range=(-1000, 1000)):
    """

    :param clip_range:
    :param original_resolution:
    :param return_original_resolution:
    :param dcm_dict:
    :param checkpoint_path_upsample: if the resolution on z is too low, upsample it.
    :param show: show information during processing
    :param batch_size:
    :return: the rescaled_ct for chest CT
    """
    raw_data_array = simple_stack_dcm_files(dcm_dict)

    raw_data_array = np.clip(raw_data_array, clip_range[0], clip_range[1])

    signal_rescaled = (raw_data_array + 600) / 1600

    if original_resolution is not None:
        resolutions = list(original_resolution)
    else:
        resolutions = get_resolution_from_dcm(dcm_dict)

    resolution_z = resolutions[2]

    if resolution_z > 2.5:
        if show:
            print("low resolution on z, upsample it")

        if not np.shape(signal_rescaled)[0: 2] == (512, 512):
            if show:
                print('original x y shape is ', np.shape(signal_rescaled)[0: 2], ', rescale to (512, 512)')

            target_shape = (512, 512, np.shape(signal_rescaled)[2])

            signal_rescaled = spatial_normalize.rescale_to_new_shape(signal_rescaled, target_shape)

        if not (5.1 > resolution_z > 4.9):
            if show:
                print("the resolution on z is not 5mm, change z resolution to 5mm")

            current_z_shape = np.shape(signal_rescaled)[2]
            ct_length_z = current_z_shape * resolution_z
            new_z_shape = round(ct_length_z / 5)

            target_shape = (512, 512, new_z_shape)

            signal_rescaled = spatial_normalize.rescale_to_new_shape(signal_rescaled, target_shape)

        up_sampled_ct = up_sample.upsample_array_z_from_5mm_to_1mm(
            signal_rescaled, checkpoint_path_upsample, batch_size)

        resolutions[2] = 1

        print("resolution before rescale:", resolutions)
        rescaled_ct = spatial_normalize.rescale_to_standard(up_sampled_ct, resolutions, change_z_resolution=False)

        if not return_original_resolution:
            return rescaled_ct
        return rescaled_ct, resolutions

    print("resolution before rescale:", resolutions)
    rescaled_ct = spatial_normalize.rescale_to_standard(signal_rescaled, resolutions, change_z_resolution=False)

    if not return_original_resolution:
        return rescaled_ct
    return rescaled_ct, resolutions


def establish_rescaled_mask(mha_path_or_mask_array, source_dcm_dict=None, resolutions=None, cast_to_binary=False):
    """

    :param mha_path_or_mask_array:
    :param source_dcm_dict:
    :param resolutions:
    :param cast_to_binary:
    :return: the mask in rescaled array
    """

    if source_dcm_dict is None:
        assert resolutions is not None and len(resolutions) == 3
    else:
        assert resolutions is None
        resolutions = get_resolution_from_dcm(source_dcm_dict)

        if resolutions[2] > 2.5:
            resolutions[2] = 1  # the rescale will upsample it to 1mm

    if type(mha_path_or_mask_array) is str:
        mask_array = simple_abstract_mha_file(mha_path_or_mask_array, cast_to_binary=False)
        mask_rescaled = spatial_normalize.rescale_to_standard(mask_array, resolutions, change_z_resolution=False)
        if cast_to_binary:
            return np.array(mask_rescaled > 0.5, 'float16')
        return mask_rescaled
    else:
        mask_array = mha_path_or_mask_array
        mask_rescaled = spatial_normalize.rescale_to_standard(mask_array, resolutions, change_z_resolution=False)
        if cast_to_binary:
            return np.array(mask_rescaled > 0.5, 'float16')
        return mask_rescaled


def undo_spatial_rescale(rescaled_array, original_dcm_dict=None, original_resolution=None, original_shape=None,
                         resolution_rescaled=(334/512, 334/512, 1)):
    """

    undo the spatial rescale, return array aligned with simple_stack_dcm_files(original_dcm_dict)

    :param rescaled_array:
    :param original_dcm_dict:
    :param original_resolution: like (0.9, 0.9, 0.4)
    :param original_shape: like (512, 512, 295)
    :param resolution_rescaled:
    :return: numpy array in float32, aligned with simple_stack_dcm_files(original_dcm_dict)
    """
    if original_resolution is None:
        original_resolution = get_resolution_from_dcm(original_dcm_dict)
        if original_resolution[2] > 2.5:
            original_resolution[2] = 1
    if original_shape is None:
        original_shape = np.shape(simple_stack_dcm_files(original_dcm_dict))

    return spatial_normalize.rescale_to_original(
        rescaled_array, resolution_rescaled, original_resolution, original_shape)


def pipeline_form_rescaled_ct(
        dict_source_file_rsna='/data_disk/RSNA-PE_dataset/rsna-str-pulmonary-embolism-detection/train',
        dict_rescaled_ct='/data_disk/RSNA-PE_dataset/rescaled_ct', fold=(0, 1)):
    scan_name_list = os.listdir(dict_source_file_rsna)[fold[0]:: fold[1]]
    fn_existing = os.listdir(dict_rescaled_ct)
    processed_count = 0
    for scan_name in scan_name_list:
        print("processing:", scan_name, processed_count, '/', len(scan_name_list))
        if scan_name + '.npz' in fn_existing:
            print("processed")
            processed_count += 1
            continue
        try:
            dict_dcm = os.path.join(dict_source_file_rsna, scan_name)
            sub_dirs_list = os.listdir(dict_dcm)
            assert len(sub_dirs_list) == 1
            dict_dcm = os.path.join(dict_dcm, sub_dirs_list[0])
            rescaled_ct = establish_rescale_chest_ct(dict_dcm)
            rescaled_ct = np.array(rescaled_ct, 'float16')
            Functions.save_np_array(dict_rescaled_ct, scan_name + '.npz', rescaled_ct, compress=True)
        except:
            print("failed")
            continue
        processed_count += 1


if __name__ == '__main__':

    pipeline_form_rescaled_ct(fold=(0, 4))
    exit()

    top_dict = '/data_disk/RSNA-PE_dataset/rsna-str-pulmonary-embolism-detection/test'
    fn_list = os.listdir(top_dict)
    for fn in fn_list:
        fn_sub_list = os.listdir(os.path.join(top_dict, fn))
        if not len(fn_sub_list) == 1:
            print(fn, len(fn_sub_list))

    exit()
    rescale_ct_ = establish_rescale_chest_ct('/data_disk/RSNA-PE_dataset/rsna-str-pulmonary-embolism-detection/'
                                             'train/1f19d8a172f2/c97f9131243e')
    print(np.shape(rescale_ct_))
