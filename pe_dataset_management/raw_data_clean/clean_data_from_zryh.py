import os
import Tool_Functions.Functions as Functions
import Tool_Functions.file_operations as file_operations
import shutil


def rename_folders_for_dataset(top_dict):

    def rename_folder(folder_name):
        if 'zryh' not in folder_name:
            if folder_name[0:10] == 'patient-id':
                new_folder_name = 'zryh' + folder_name[10::]
            else:
                new_folder_name = 'zryh-' + folder_name
        else:
            new_folder_name = folder_name

        return new_folder_name

    folder_name_list = os.listdir(top_dict)
    for name in folder_name_list:
        new_name = rename_folder(name)

        file_operations.rename_file_or_folder_name(os.path.join(top_dict, name), os.path.join(top_dict, new_name))


def clean_scan_folder(dict_contain_dcm):
    """

    dcm_dict is considered as the folder with the most files

    :param dict_contain_dcm:
    :return:
    """
    directory_list = file_operations.extract_all_directory(dict_contain_dcm, complete_tree=True)
    name_file_count_list = []
    if len(directory_list) == 1:
        return None

    for directory in directory_list:
        name_file_count_list.append((directory, len(os.listdir(directory))))

    def sort_func(a, b):
        if a[1] < b[1]:
            return 1
        return -1

    name_file_count_list = Functions.customized_sort(name_file_count_list, sort_func)

    dcm_dict = name_file_count_list[0][0]

    path_need_remove_1 = os.path.join(dcm_dict, 'VERSION')
    if os.path.exists(path_need_remove_1):
        os.remove(path_need_remove_1)

    temp_dcm_folder = os.path.join(dict_contain_dcm, 'dcm_folder')
    file_operations.rename_file_or_folder_name(dcm_dict, temp_dcm_folder)

    name_list = os.listdir(dict_contain_dcm)
    for name in name_list:
        if not name == 'dcm_folder':
            file_operations.remove_path_or_directory(os.path.join(dict_contain_dcm, name))

    dcm_name_list = os.listdir(temp_dcm_folder)
    for dcm_name in dcm_name_list:
        shutil.move(os.path.join(temp_dcm_folder, dcm_name), os.path.join(dict_contain_dcm, dcm_name))

    file_operations.remove_path_or_directory(temp_dcm_folder)

    return None


def clean_patient_name(top_dict_pair):
    if os.path.exists(os.path.join(top_dict_pair, 'non')):
        file_operations.rename_file_or_folder_name(os.path.join(top_dict_pair, 'non'),
                                                   os.path.join(top_dict_pair, 'non-contrast'))
    clean_scan_folder(os.path.join(top_dict_pair, 'CTA'))
    clean_scan_folder(os.path.join(top_dict_pair, 'non-contrast'))


def clean_dataset(dataset_dict):

    dataset_a = os.path.join(dataset_dict, 'may_not_be_PE')
    dataset_b = os.path.join(dataset_dict, 'should_be_PE')

    name_list = os.listdir(dataset_a)
    for name in name_list:
        clean_patient_name(os.path.join(dataset_a, name))

    name_list = os.listdir(dataset_b)
    for name in name_list:
        clean_patient_name(os.path.join(dataset_b, name))


if __name__ == '__main__':
    clean_patient_name('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_High_Quality/patient-zr-0003')
    exit()

    clean_dataset('/data_disk/CTA-CT_paired-dataset/transfer/CTPA_zryh/hupianpian')
    clean_dataset('/data_disk/CTA-CT_paired-dataset/transfer/CTPA_zryh/lv kuan')
    clean_dataset('/data_disk/CTA-CT_paired-dataset/transfer/CTPA_zryh/yu hongwei')

