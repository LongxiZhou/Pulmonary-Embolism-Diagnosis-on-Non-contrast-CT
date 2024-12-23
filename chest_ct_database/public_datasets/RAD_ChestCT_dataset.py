"""
dataset URL is https://zenodo.org/record/6406114#.Ytl6OXbMLAQ)

"""
import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.spatial_normalize as spatial_normalize


def read_the_dcm_info(name_info_dict=None, scan_name=None, attribute_name=None):
    if attribute_name is not None:
        assert scan_name is not None

    if name_info_dict is None:
        name_info_dict = Functions.pickle_load_object(
            '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/scan-name_dcm-info_dict.pickle')

    if scan_name is None:
        return name_info_dict
    if attribute_name is None:
        return name_info_dict[scan_name]
    return name_info_dict[scan_name][attribute_name]


def rad_dataset_ct_to_rescaled_ct(array_of_rad_dataset):
    """
    the array in the RAD-ChestCT_dataset with resolution [0.8, 0.8, 0.8], no padding, values in HU units
    :param array_of_rad_dataset:
    :return: rescaled_array, the final_resolution for rescaled_array
    """
    temp_array = (array_of_rad_dataset + 600) / 1600
    temp_array = np.swapaxes(temp_array, 0, 2)
    temp_array = np.swapaxes(temp_array, 0, 1)
    temp_array = np.flip(temp_array, 2)

    rescaled_ct, final_resolution = \
        spatial_normalize.rescale_to_standard(
            temp_array, [334/512, 334/512, 0.8], change_z_resolution=True, return_final_resolution=True)

    return rescaled_ct, final_resolution


def load_func_for_ct(path_stack_ct_in_rad_format):
    rad_ct = np.load(path_stack_ct_in_rad_format)['ct']
    rescaled_ct, _ = rad_dataset_ct_to_rescaled_ct(rad_ct)
    return rescaled_ct


def rescaled_array_to_rad_array(rescaled_array, resolution_rescaled_ct, original_shape, is_mask=True):
    """

    :param original_shape: the original shape for the array in RAD-ChestCT_dataset
    :param resolution_rescaled_ct: sometimes the range for scan is very very large, so 512 mm cannot hold and changed
    the z resolution for rescaled_ct
    :param rescaled_array:
    :param is_mask: if True, does not change the CT value.
    :return: shape and resolution and value in rad dataset format
    """

    original_shape = [original_shape[1], original_shape[2], original_shape[0]]

    temp_array = spatial_normalize.rescale_to_original(rescaled_array, resolution_rescaled_ct, (334/512, 334/512, 0.8),
                                                       original_shape)

    temp_array = np.flip(temp_array, 2)
    temp_array = np.swapaxes(temp_array, 0, 1)
    temp_array = np.swapaxes(temp_array, 0, 2)

    if not is_mask:
        temp_array = temp_array * 1600 - 600

    return temp_array


def segment_semantics(fold=(0, 4)):
    from pulmonary_nodules.predict_pipeline import rescaled_ct_to_semantic_seg
    top_dict_data = '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/stack_ct_rad_format/'
    top_dict_save = '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/semantic_in_rescaled_ct/'
    rescaled_ct_to_semantic_seg(top_dict_data, top_dict_save, artery_vein=False, batch_size=4, fold=fold,
                                load_func=load_func_for_ct)


if __name__ == '__main__':
    import os
    source_dict = '/data_disk/RAD-ChestCT_dataset/stack_ct_rad_format/'
    save_dict = '/data_disk/RAD-ChestCT_dataset/rescaled_ct/'
    fn_list = os.listdir('/data_disk/RAD-ChestCT_dataset/stack_ct_rad_format/')

    for fn in fn_list:
        old_array = np.load()

    exit()

    segment_semantics((0, 5))

    exit()

    array = np.load('/media/zhoul0a/New Volume/RAD-ChestCT_dataset/trn00139.npz')['ct']

    rescaled_array, final_resolution = rad_dataset_ct_to_rescaled_ct(array)

    array_2 = rescaled_array_to_rad_array(rescaled_array, final_resolution, np.shape(array), False)

    print(np.sum(np.abs(array_2 - array)))

    Functions.image_show(array[200, :, :])
    Functions.image_show(array_2[200, :, :])

    print(np.shape(array_2))

    exit()

    array = np.swapaxes(array, 0, 2)
    array = np.swapaxes(array, 0, 1)
    array = np.flip(array, 2)

    print(np.shape(array))
    info_dict = Functions.pickle_load_object(
        '/media/zhoul0a/New Volume/RAD-ChestCT_dataset/scan-name_dcm-info_dict.pickle')

    print(read_the_dcm_info(info_dict, 'trn00139'))

    for i in range(100, 450, 5):
        Functions.image_show(np.clip(array[:, :, i], -1000, 200), gray=True)
