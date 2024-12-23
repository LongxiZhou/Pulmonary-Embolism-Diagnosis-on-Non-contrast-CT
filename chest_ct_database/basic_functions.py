import os
import warnings
import shutil

from Tool_Functions.file_operations import extract_all_file_path


def extract_absolute_dirs_sub_dataset(top_dict_source, list_dict_sub_dataset=None, in_recursion=False):

    # return list like [os.path.join(top_dict_rescaled_ct, sub_dataset_dirs), ]
    if not os.path.exists(top_dict_source):
        return []
    if not os.path.isdir(top_dict_source):
        raise ValueError("Is not directory:", top_dict_source)
    assert os.path.isdir(top_dict_source)

    if not in_recursion:
        list_dict_sub_dataset = []
    else:
        assert list_dict_sub_dataset is not None

    list_sub_dirs = os.listdir(top_dict_source)

    if len(list_sub_dirs) == 0:
        list_dict_sub_dataset.append(top_dict_source)
    elif not os.path.isdir(os.path.join(top_dict_source, list_sub_dirs[0])):
        list_dict_sub_dataset.append(top_dict_source)
    else:
        for sub_dir in list_sub_dirs:

            new_top_dict_source = os.path.join(top_dict_source, sub_dir)

            extract_absolute_dirs_sub_dataset(new_top_dict_source, list_dict_sub_dataset, True)

    return list_dict_sub_dataset


def extract_relative_dirs_sub_dataset(top_dict_source):

    # return list like ['healthy_people/', 'COVID-19/center_1/', "COVID-19/center_2", ...]

    list_dataset_dict = extract_absolute_dirs_sub_dataset(top_dict_source)

    list_sub_dirs = []
    for dataset_dict in list_dataset_dict:
        item = dataset_dict[len(top_dict_source)::]
        if len(item) == 0:
            list_sub_dirs.append(item)
            continue
        if item[0] == '/':
            item = item[1::]
        list_sub_dirs.append(item)

    return list_sub_dirs


def merge_dicts(list_of_dict, check_overlap_keys=True):
    """

    merge several dictionary into one. key should be unique for all_file dicts

    :param list_of_dict:
    :param check_overlap_keys: if True, require overlap key have same value
    :return:
    """
    assert len(list_of_dict) > 0
    merged_dict = {}
    key_set_merged = set()

    for item in list_of_dict:
        key_set = set(item.keys())

        if check_overlap_keys:
            overlap_keys = key_set_merged & key_set
            if len(overlap_keys) > 0:
                for key in overlap_keys:
                    if not merged_dict[key] == item[key]:
                        print("two value in same key", key)
                        print("value 1:", merged_dict[key])
                        print("value 2:", item[key])
                        raise ValueError("dicts have same key with different values")

        key_set_merged = key_set_merged | key_set
        for key, value in item.items():
            merged_dict[key] = value

    return merged_dict


def merge_features(top_dict_source, list_top_dict_feature, top_dict_merged_feature, policy='move'):
    """

    we have two features:
    feature_1/sub_dataset_dict../feature_files
    feature_2/sub_dataset_dict../feature_files

    merge to:
    new_feature_name/sub_dataset_dict../feature_files_merged

    :param top_dict_source: we can extract all_file dataset dirs
    :param list_top_dict_feature: [top_dict_1, top_dict_2, ...]
    :param top_dict_merged_feature: the dict for the new merged feature
    :param policy: 'move' or 'copy'
    :return:
    """
    list_sub_dataset = extract_relative_dirs_sub_dataset(top_dict_source)

    list_feature_name_set = []

    sub_feature_name_set = set()
    for top_dict_feature in list_top_dict_feature:
        for sub_dataset in list_sub_dataset:
            directory_feature = os.path.join(top_dict_feature, sub_dataset)
            feature_name_list_sub = os.listdir(directory_feature)
            if os.path.isfile(os.path.join(directory_feature, feature_name_list_sub[0])):
                continue
            feature_name_set_sub = set(feature_name_list_sub)
            assert len(feature_name_set_sub) == len(feature_name_list_sub)
            sub_feature_name_set = sub_feature_name_set | feature_name_set_sub

        if len(sub_feature_name_set) > 0:
            print("feature:", top_dict_feature, '\nhas the following sub feature files:')
            print(sub_feature_name_set, '\n')
        else:
            print("feature:", top_dict_feature, '\nhas no sub features')
        list_feature_name_set.append(sub_feature_name_set)
        sub_feature_name_set = set()

    if not os.path.exists(top_dict_merged_feature):
        os.makedirs(top_dict_merged_feature)

    num_features = len(list_top_dict_feature)

    for feature_index in range(num_features):
        top_dict_feature = list_top_dict_feature[feature_index]
        print("\nmerging feature:", top_dict_feature)
        print("feature merged:", feature_index, '/', num_features)
        sub_feature_name_set = list_feature_name_set[feature_index]

        if len(sub_feature_name_set) > 0:
            processed_dataset_count = 0
            for sub_dataset in list_sub_dataset:
                print("extract from dataset:", sub_dataset, processed_dataset_count, '/', len(list_sub_dataset))
                for sub_feature_name in sub_feature_name_set:
                    source_dict = os.path.join(top_dict_feature, sub_dataset, sub_feature_name)
                    if not os.path.exists(source_dict):
                        continue
                    destiny_dict = os.path.join(top_dict_merged_feature, sub_dataset, sub_feature_name)
                    if not os.path.exists(destiny_dict):
                        os.makedirs(destiny_dict)

                    file_list = os.listdir(source_dict)
                    for file_name in file_list:
                        source_path = os.path.join(source_dict, file_name)
                        destiny_path = os.path.join(destiny_dict, file_name)
                        if os.path.exists(destiny_path):
                            continue
                        if os.path.exists(source_path):
                            if policy == 'copy':
                                shutil.copyfile(source_path, destiny_path)
                            if policy == 'move':
                                shutil.move(source_path, destiny_path)
                processed_dataset_count += 1
        else:
            processed_dataset_count = 0
            for sub_dataset in list_sub_dataset:
                print("dataset:", sub_dataset, processed_dataset_count, '/', len(list_sub_dataset))
                if top_dict_feature[-1] == '/':
                    top_dict_feature = top_dict_feature[:-1]
                sub_feature_name = top_dict_feature.split('/')[-1]

                new_sub_feature_dir = os.path.join(top_dict_merged_feature, sub_dataset, sub_feature_name)
                if not os.path.exists(new_sub_feature_dir):
                    os.makedirs(new_sub_feature_dir)

                file_list = os.listdir(os.path.join(top_dict_feature, sub_dataset))
                for single_file_name in file_list:
                    source_path = os.path.join(top_dict_feature, sub_dataset, single_file_name)
                    destiny_path = os.path.join(
                        top_dict_merged_feature, sub_dataset, sub_feature_name, single_file_name)
                    if os.path.exists(destiny_path):
                        continue
                    if os.path.exists(source_path):
                        if policy == 'copy':
                            shutil.copyfile(source_path, destiny_path)
                        if policy == 'move':
                            shutil.move(source_path, destiny_path)
                processed_dataset_count += 1


def rename_sub_feature_name(top_dict_source, top_dict_feature, name_sub_feature, new_name_sub_feature):
    """

    :param top_dict_source: to get the list of sub_dataset
    :param top_dict_feature: feature top dict for rename
    :param name_sub_feature: a string
    :param new_name_sub_feature: a string
    :return:
    """
    list_sub_dataset = extract_relative_dirs_sub_dataset(top_dict_source)

    for sub_dataset in list_sub_dataset:
        original_dict = os.path.join(top_dict_feature, sub_dataset, name_sub_feature)
        if os.path.exists(original_dict):
            assert os.path.isdir(original_dict)
            new_dict = os.path.join(top_dict_feature, sub_dataset, new_name_sub_feature)
            os.rename(original_dict, new_dict)
        else:
            warnings.warn("sub feature not found.")
            print("Sub feature name \"", name_sub_feature, "\" not found in", sub_dataset)


def delete_sub_feature(top_dict_source, top_dict_feature, name_sub_feature):
    """

        :param top_dict_source: to get the list of sub_dataset
        :param top_dict_feature: feature top dict for rename
        :param name_sub_feature: a string
        :return:
        """
    list_sub_dataset = extract_relative_dirs_sub_dataset(top_dict_source)

    for sub_dataset in list_sub_dataset:
        sub_feature_dict = os.path.join(top_dict_feature, sub_dataset, name_sub_feature)
        if os.path.exists(sub_feature_dict):
            assert os.path.isdir(sub_feature_dict)
            shutil.rmtree(sub_feature_dict)
        else:
            warnings.warn("sub feature not found.")
            print("Sub feature name \"", name_sub_feature, "\" not found in", sub_dataset)


if __name__ == '__main__':

    print(extract_relative_dirs_sub_dataset('/data_disk/rescaled_ct_and_semantics'))
    exit()

    path_list_ = extract_all_file_path('/data_disk/pulmonary_embolism/simulated_lesions', '.docx')
    print(len(path_list_))
    for item_ in path_list_:
        print(item_)
    exit()
    print(extract_relative_dirs_sub_dataset('/data_disk/rescaled_ct_and_semantics/reports'))
    exit()

    delete_sub_feature('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/',
                       '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/depth_and_center-line/',
                       'depth_array')

    exit()

    rename_sub_feature_name('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/',
                            '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/depth_and_center-line/',
                            'center_line_mask', 'blood_center_line')

    exit()

    merge_features('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/',
                   ['/media/zhoul0a/New Volume/rescaled_ct_and_semantics/airway_center_line/',
                    '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/blood_depth_and_center-line/'],
                   '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/depth_and_center-line/', 'copy')
    exit()
    dict_1 = {1: '33', 2: '44', 'hh': 4}
    dict_2 = {111: '332222', 2333: '433334', 'hh': 4444}

    print(merge_dicts([dict_1, dict_2]))
