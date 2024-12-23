import pydicom
import SimpleITK as sitk
import os
import format_convert.spatial_normalize as spatial_normalize
import Tool_Functions.Functions as Functions
import lung_atlas.upsample_ct_z.predict_up_sampled as up_sample
import numpy as np


def simple_stack_dcm_files(dcm_dict):
    reader = sitk.ImageSeriesReader()

    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dict)

    reader.SetFileNames(dcm_series)

    img = reader.Execute()

    img_array = sitk.GetArrayFromImage(img)  # z y x

    img_array = np.swapaxes(img_array, 0, 2)

    img_array = np.swapaxes(img_array, 0, 1)

    return img_array


def simple_abstract_mha_file(mha_path, cast_to_binary=False):
    ar = sitk.ReadImage(mha_path)

    mask = sitk.GetArrayFromImage(ar)  # z y x

    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)

    if cast_to_binary:
        mask = np.array(mask > 0.5, 'float32')

    return mask  # (x, y, z)


def get_resolution_from_dcm(dcm_dict, show=True):
    dcm_file_names = os.listdir(dcm_dict)
    num_slices = len(dcm_file_names)
    assert num_slices > 1

    if show:
        print("number_dcm_files:", num_slices)

    first_content = pydicom.read_file(os.path.join(dcm_dict, dcm_file_names[0]))

    resolutions_xy = first_content.PixelSpacing

    slice_id_z_location_list = []  # each item is (slice_id, z_location)

    for file_name in dcm_file_names:

        dcm_file = pydicom.read_file(os.path.join(dcm_dict, file_name))

        try:
            slice_id = float(dcm_file['InstanceNumber'].value)
            z_location = float(dcm_file['SliceLocation'].value)
            slice_id_z_location_list.append((slice_id, z_location))
        except:
            print('file', file_name, 'cannot extract instance number or slice location, and no SliceThickness')
            os.remove(os.path.join(dcm_dict, file_name))

    assert len(slice_id_z_location_list) >= 2

    def sort_list(item_a, item_b):
        if item_a[0] > item_b[0]:
            return 1
        return -1

    slice_id_z_location_list = Functions.customized_sort(slice_id_z_location_list, sort_list, reverse=False)

    resolution_by_sample_set = set()
    max_slice = slice_id_z_location_list[0][0]
    min_slice = slice_id_z_location_list[0][0]
    index_max = 0
    index_min = 0

    for index in range(1, len(slice_id_z_location_list)):

        if slice_id_z_location_list[index][0] > max_slice:
            max_slice = slice_id_z_location_list[index][0]
            index_max = index
        if slice_id_z_location_list[index][0] < min_slice:
            min_slice = slice_id_z_location_list[index][0]
            index_min = index

        interval_slices = abs(slice_id_z_location_list[index][0] - slice_id_z_location_list[index - 1][0])
        assert interval_slices > 0
        z_distance = abs(slice_id_z_location_list[index][1] - slice_id_z_location_list[index - 1][1])
        resolution_z = z_distance / interval_slices
        resolution_by_sample_set.add(resolution_z)

    assert not max_slice == min_slice
    interval_series = max_slice - min_slice
    length_series = abs(slice_id_z_location_list[index_max][1] - slice_id_z_location_list[index_min][1])
    resolution_z = length_series / interval_series

    if len(resolution_by_sample_set) > 1 and show:
        print("different slice thickness on z:", resolution_by_sample_set)
        assert min(resolution_by_sample_set) > 0
        assert max(resolution_by_sample_set) / min(resolution_by_sample_set) < 1.05

    resolutions = [float(resolutions_xy[0]), float(resolutions_xy[1]), resolution_z]

    if show:
        print('the resolution for x, y, z in mm:', resolutions)

    return resolutions


def establish_rescale_chest_ct(dcm_dict, checkpoint_path_upsample=None, show=True, batch_size=2,
                               return_original_resolution=False, original_resolution=None):
    """

    :param original_resolution:
    :param return_original_resolution:
    :param dcm_dict:
    :param checkpoint_path_upsample: if the resolution on z is too low, upsample it.
    :param show: show information during processing
    :param batch_size:
    :return: the rescaled_ct for chest CT
    """
    raw_data_array = simple_stack_dcm_files(dcm_dict)

    raw_data_array = np.clip(raw_data_array, -1000, 2000)

    signal_rescaled = (raw_data_array + 600) / 1600

    if original_resolution is not None:
        resolutions = list(original_resolution)
    else:
        resolutions = get_resolution_from_dcm(dcm_dict, show=show)

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

        rescaled_ct = spatial_normalize.rescale_to_standard(up_sampled_ct, resolutions, change_z_resolution=False)

        if not return_original_resolution:
            return rescaled_ct
        return rescaled_ct, resolutions

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
        original_resolution = get_resolution_from_dcm(original_dcm_dict, show=False)
        if original_resolution[2] > 2.5:
            original_resolution[2] = 1
    if original_shape is None:
        original_shape = np.shape(simple_stack_dcm_files(original_dcm_dict))

    return spatial_normalize.rescale_to_original(
        rescaled_array, resolution_rescaled, original_resolution, original_shape)


if __name__ == '__main__':

    array = simple_stack_dcm_files('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_23-12-10/N324-N435/N435/CTA')
    Functions.image_show(np.clip(array[:, :, 150], -1000, 400), gray=True)
    exit()
    Functions.load_dicom('/media/zhoul0a/My Passport/3T/6511278/ser9/ser009img00001.dcm', show=True)
