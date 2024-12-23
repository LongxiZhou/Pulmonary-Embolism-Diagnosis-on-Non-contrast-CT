import os
from chest_ct_database.basic_functions import extract_absolute_dirs_sub_dataset


"""
"top_dict_rescaled_ct": get the sub directory for each dataset, i.e., directory only contains files, like "healthy/xwzc"
"list_top_dict_reference": materials for calculate new feature, e.g., os.path.join("top_dict_reference", sub directory) 
provide a material.
"top_dict_save": save new feature to os.path.join("top_dict_save", sub directory) 

func_pre_load(), 
returns a dict, stores loaded models, then pass to func_file_operations **kwargs

func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, **kwargs=func_pre_load())
returns an object "feature_package", then pass to func_file_save

func_file_save(save_dict, file_name, "feature_package")

func_check_processed(save_dict, file_name)
return True for processed, False for not processed
"""


def func_add_feature(top_dict_source, list_top_dict_reference, top_dict_save, func_file_operation, func_file_save,
                     func_pre_load=None, func_check_processed=None, fold=(0, 1)):
    """
    apply func_for_file for every file in the top_dict_rescaled_ct, and then save it with func_file_save

    :param top_dict_source: to get the path for all_file instance (files), like ./our_dataset/rescaled_ct/
    :param list_top_dict_reference: features of instance (file)
    :param top_dict_save: like ./our_dataset/semantics/
    :param func_file_operation: func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name, **kwargs)
    :param func_file_save: func_file_save(save_dict, file_name, return_of_func_file_operation)
    :param func_pre_load: return a dictionary, pre-load models, arrays for func_file_operation
    :param func_check_processed: func_check_processed(save_dict, file_name)
    :param fold
    :return: None
    """
    if not top_dict_source[-1] == '/':
        top_dict_source = top_dict_source + '/'

    if func_pre_load is not None:
        dict_pre_load = func_pre_load()
    else:
        dict_pre_load = None

    list_dataset_dict = extract_absolute_dirs_sub_dataset(top_dict_source)
    list_dataset_dict.sort()
    print("there are", len(list_dataset_dict), 'dataset under', top_dict_source)
    print("fold:", fold)

    list_sub_dirs = []
    for dataset_dict in list_dataset_dict:
        list_sub_dirs.append(dataset_dict[len(top_dict_source)::])

    for dataset_sub_dir in list_sub_dirs:  # dataset_sub_dir like 'COVID-19/healthy/four_center'
        print("##########################")
        print("processing dataset:", dataset_sub_dir)
        save_dict = os.path.join(top_dict_save, dataset_sub_dir)
        print("saving new feature to:", save_dict)
        print("##########################")

        dataset_dict = os.path.join(top_dict_source, dataset_sub_dir)
        file_name_list = os.listdir(dataset_dict)
        file_name_list = down_sample_path_list_according_to_sor_sum(file_name_list, fold=fold)

        total_file = len(file_name_list)
        processed_count = 0

        for file_name in file_name_list:
            print("processing:", file_name, processed_count, '/', total_file)
            if func_check_processed is not None:
                if func_check_processed(save_dict, file_name):
                    print("processed")
                    processed_count += 1
                    continue

            if dict_pre_load is None:
                feature_package = func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name)
            else:
                feature_package = func_file_operation(
                    list_top_dict_reference, dataset_sub_dir, file_name, **dict_pre_load)

            func_file_save(save_dict, file_name, feature_package)
            processed_count += 1

    return None


def get_ord_sum(string):
    ord_sum = 0
    for char in string:
        ord_sum += ord(char)
    return ord_sum


def down_sample_path_list_according_to_sor_sum(path_list, fold=(0, 1)):
    path_list.sort()
    fold = list(fold)
    reverse_fold = False
    if fold[0] < 0:
        reverse_fold = True
    if fold[0] == -fold[1]:
        fold[0] = 0
    new_list = []
    for path in path_list:
        if get_ord_sum(path) % fold[1] == abs(fold[0]):
            new_list.append(path)
    if reverse_fold:
        new_list.reverse()
    return new_list


if __name__ == '__main__':
    list_dict = extract_absolute_dirs_sub_dataset(
        '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/denoise_ct_float16/')
    for item in list_dict:
        print(item)
    exit()

    func_add_feature('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct', None, None, None, None)
    exit()
