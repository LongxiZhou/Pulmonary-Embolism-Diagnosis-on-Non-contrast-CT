import os
import Tool_Functions.file_operations as file_operations


def clean_dataset(top_dict, func_clean=None):
    list_all_path = file_operations.extract_all_file_path(top_dict)

    def default_func_clean(path):
        file_name = path.split('/')[-1]
        if '._' in file_name:
            return True
        if '.DS_Store' in file_name:
            return True

        return False

    if func_clean is None:
        func_clean = default_func_clean

    for file_path in list_all_path:
        if func_clean(file_path):
            os.remove(file_path)

    return None


def clean_case(dict_case):
    file_name_list = os.listdir(dict_case)
    if 'StudyInfo.dat' in file_name_list:
        os.remove(os.path.join(dict_case, 'StudyInfo.dat'))
    if os.path.exists(os.path.join(dict_case, 'non-contrast')):
        assert not os.path.exists(os.path.join(dict_case, 'NON'))
        return
    os.rename(os.path.join(dict_case, 'NON'), os.path.join(dict_case, 'non-contrast'))


def clean_nj_dataset(top_dict):
    clean_dataset(top_dict)
    case_list = os.listdir(top_dict)
    for case in case_list:
        print("clean:", case)
        clean_case(os.path.join(top_dict, case))


if __name__ == '__main__':
    clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_10/NJ1001-1100')
    clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_10/NJ1101-1202')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_04/NJ0901-1000')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_04/NJ0801-900')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_04/NJ0701-800')
    exit()
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_04/NJ0601-700')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_03_26/NJ0501-600')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_03_26/NJ0401-500')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_03_21/NJ0301-400')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_03_11/NJ0101-300')
    # clean_nj_dataset('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_02_26/NJ0001-0100')
