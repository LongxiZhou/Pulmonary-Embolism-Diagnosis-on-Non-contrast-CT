"""
Some dataset may share same scan. This program synchronize the features among these scan

source: will not be modified if bi_direction is set as False
target: if find a corresponding feature in source, and is not exist in target or newer (optional), then copy to target.

"""
import os
import Tool_Functions.Functions as Functions
import chest_ct_database.basic_functions as basic_functions


def get_overlap_name_field(list_of_folder_path_rescaled_ct_source, list_of_folder_path_rescaled_ct_target):
    """

    rescaled_ct defines the dataset.
    return a set, containing overlap scan names (stripped suffix like .npz)

    :param list_of_folder_path_rescaled_ct_source:
    :param list_of_folder_path_rescaled_ct_target:
    :return:
    """
    if type(list_of_folder_path_rescaled_ct_source) is str:
        list_of_folder_path_rescaled_ct_source = [list_of_folder_path_rescaled_ct_source]
    else:
        assert len(list_of_folder_path_rescaled_ct_source) > 0
        assert type(list_of_folder_path_rescaled_ct_source[0]) is str
    if type(list_of_folder_path_rescaled_ct_target) is str:
        list_of_folder_path_rescaled_ct_target = [list_of_folder_path_rescaled_ct_target]
    else:
        assert len(list_of_folder_path_rescaled_ct_target) > 0
        assert type(list_of_folder_path_rescaled_ct_target[0]) is str

    all_fn_source = []
    for folder_path in list_of_folder_path_rescaled_ct_source:
        all_fn_source = all_fn_source + os.listdir(folder_path)

    all_fn_target = []
    for folder_path in list_of_folder_path_rescaled_ct_target:
        all_fn_target = all_fn_target + os.listdir(folder_path)

    fn_list_source = []
    for fn in all_fn_source:
        fn_list_source.append(Functions.strip_suffix(fn))
    fn_list_target = []
    for fn in all_fn_target:
        fn_list_target.append(Functions.strip_suffix(fn))

    assert len(fn_list_source) == len(set(fn_list_source))
    assert len(fn_list_target) == len(set(fn_list_target))

    return set(fn_list_source) & set(fn_list_target)


def get_overlap_name_field_database(top_dict_source, top_dict_target):
    """

    find same scan name in two database

    :param top_dict_source:
    :param top_dict_target:
    :return:
    """
    list_of_folder_path_source = basic_functions.extract_absolute_dirs_sub_dataset(top_dict_source)
    list_of_folder_path_target = basic_functions.extract_absolute_dirs_sub_dataset(top_dict_target)
    return get_overlap_name_field(list_of_folder_path_source, list_of_folder_path_target)


def is_source_newer(source_path, target_path):
    """

    :param source_path:
    :param target_path:
    :return: True or False
    """
    # check not folder
    assert os.path.isfile(source_path) and os.path.isfile(target_path)
    time_modified_source = os.path.getmtime(source_path)  # sec passed from 1970 0101
    time_modified_target = os.path.getmtime(target_path)

    if time_modified_source - time_modified_target > 0:
        return True
    return False


def synchronize_feature_file(feature_path_source, feature_path_target, update=False):
    """

    :param update:
    :param feature_path_source:
    :param feature_path_target:
    :return:
    """

    if not os.path.exists(feature_path_source):
        return None

    if not os.path.exists(feature_path_target):
        print('\ncopying file from:', feature_path_source,)
        print('to:', feature_path_target)
        Functions.copy_file(feature_path_source, feature_path_target)

    if update and is_source_newer(feature_path_source, feature_path_target):
        print('\ncopying file from:', feature_path_source, )
        print('to:', feature_path_target)
        Functions.copy_file(feature_path_source, feature_path_target)


def synchronize_feature(feature_dict_source, feature_dict_target, overlap_name_field, bi_direction=False,
                        update=False):

    if bi_direction:
        synchronize_feature(feature_dict_target, feature_dict_source, overlap_name_field, bi_direction=False,
                            update=update)

    if not os.path.isdir(feature_dict_source):
        return None

    list_sub_feature_source = basic_functions.extract_relative_dirs_sub_dataset(feature_dict_source)

    for sub_feature in list_sub_feature_source:

        sub_feature_dict_source = os.path.join(feature_dict_source, sub_feature)
        sub_feature_dict_target = os.path.join(feature_dict_target, sub_feature)

        synchronize_feature_sub_dataset(
            sub_feature_dict_source, sub_feature_dict_target, overlap_name_field, bi_direction)


def synchronize_feature_sub_dataset(sub_feature_dict_source, sub_feature_dict_target,
                                    overlap_name_field, bi_direction=False, update=False):
    """

    require sub_feature_dict contain NO folder.

    :param update: over-write if source is newer
    :param sub_feature_dict_source:
    :param sub_feature_dict_target:
    :param overlap_name_field:
    :param bi_direction:
    :return:
    """

    if not os.path.exists(sub_feature_dict_source):
        return None
    if not os.path.exists(sub_feature_dict_target):
        os.makedirs(sub_feature_dict_target)

    if bi_direction is True:
        synchronize_feature_sub_dataset(
            sub_feature_dict_target, sub_feature_dict_source, overlap_name_field, bi_direction=False, update=update)

    feature_fn_list_source = os.listdir(sub_feature_dict_source)
    if len(feature_fn_list_source) == 0:
        return None

    feature_suffix = Functions.get_suffix(feature_fn_list_source[0])
    for scan_name in overlap_name_field:

        feature_name = scan_name + feature_suffix

        feature_path_source = os.path.join(sub_feature_dict_source, feature_name)
        feature_path_target = os.path.join(sub_feature_dict_target, feature_name)

        synchronize_feature_file(feature_path_source, feature_path_target, update=update)


def synchronize_cta_related_features(top_dict_source, top_dict_target, bi_direction=False):

    """

    :param top_dict_source: directory for rescaled-ct or rescaled-ct_denoise
    :param top_dict_target: directory for rescaled-ct or rescaled-ct_denoise
    :param bi_direction:
    :return:
    """
    overlap_name_field = get_overlap_name_field_database(top_dict_source, top_dict_target)

    source_father_dict = Functions.get_father_dict(top_dict_source)
    target_father_dict = Functions.get_father_dict(top_dict_target)

    feature_name_list = ['depth_and_center-line', 'secondary_semantics', 'semantics']
    for feature_name in feature_name_list:
        feature_dict_source = os.path.join(source_father_dict, feature_name)
        feature_dict_target = os.path.join(target_father_dict, feature_name)
        synchronize_feature(feature_dict_source, feature_dict_target, overlap_name_field, bi_direction=bi_direction)


if __name__ == '__main__':

    synchronize_cta_related_features(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/simulated_non_contrast/rescaled_ct-denoise',
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/simulated_non_contrast/rescaled_ct-denoise',
        bi_direction=True
    )
    exit()

    overlap_name = get_overlap_name_field_database(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise',
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/rescaled_ct-denoise')
    print(len(overlap_name))
    print(overlap_name)
