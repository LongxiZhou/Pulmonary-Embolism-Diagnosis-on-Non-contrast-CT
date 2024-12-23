"""
given a folder
remove "DICOMDIR", "LOCKFILE", "VERSION", ".DS_Store"
clean directory to let folder -> dcm_files

assume:
the folder contains the most files is the dcm folder, delete the other folders.
"""


import Tool_Functions.file_operations as file_operations
import os


bad_name_list = ["DICOMDIR", "LOCKFILE", "VERSION"]
bad_start_list = ["._"]
bad_end_list = []


def remove_bad_files(top_dict_folder, show=True):

    list_file_path_bad_start = []
    for bad_start in bad_start_list:
        list_file_path_bad_start = list_file_path_bad_start + \
                                   file_operations.extract_all_file_path(top_dict_folder, start_with=bad_start)
    for path in list_file_path_bad_start:
        file_operations.remove_path_or_directory(path, show=show)

    list_file_path_bad_end = []
    for bad_end in bad_end_list:
        list_file_path_bad_end = list_file_path_bad_end + \
                                   file_operations.extract_all_file_path(top_dict_folder, end_with=bad_end)
    for path in list_file_path_bad_end:
        file_operations.remove_path_or_directory(path, show=show)

    list_all_file_path = file_operations.extract_all_file_path(top_dict_folder)
    for path in list_all_file_path:
        name_file = path.split('/')[-1]
        if name_file in bad_name_list:
            file_operations.remove_path_or_directory(path, show=show)


def clean_folder_structure(top_dict_folder, show=True):
    list_all_file_path = file_operations.extract_all_file_path(top_dict_folder)
    assert len(list_all_file_path) > 0

    # all file should under the same folder
    father_dict = file_operations.get_father_dict(list_all_file_path[0])
    for path in list_all_file_path[1:]:
        if not file_operations.get_father_dict(path) == father_dict:
            print(path, father_dict)
            raise ValueError

    if father_dict == top_dict_folder:
        # there may be some empty folders, remove then
        file_operations.remove_empty_folders(top_dict_folder, remove_self=False, show=show)
        return

    for path in list_all_file_path:
        name_file = path.split('/')[-1]
        new_path = os.path.join(top_dict_folder, name_file)
        file_operations.move_file_or_dir(path, new_path, show=show)

    # there may be some empty folders, remove them
    file_operations.remove_empty_folders(top_dict_folder, remove_self=False, show=show)


def clean_scan_folder(scan_folder, show=True):
    assert os.path.exists(scan_folder)

    remove_bad_files(scan_folder, show=show)
    clean_folder_structure(scan_folder, show=show)


def clean_paired_folder(top_dict_pair, show=True):
    assert os.path.exists(top_dict_pair)

    remove_bad_files(top_dict_pair, show=show)

    scan_folder_name_list = os.listdir(top_dict_pair)
    assert len(scan_folder_name_list) == 2

    for scan_folder_name in scan_folder_name_list:
        absolute_path = os.path.join(top_dict_pair, scan_folder_name)
        if "CTA" in scan_folder_name:
            file_operations.rename_file_or_folder_name(absolute_path, "CTA", show=show)
        if "non" in scan_folder_name:
            file_operations.rename_file_or_folder_name(absolute_path, "non-contrast", show=show)

    folder_cta = os.path.join(top_dict_pair, "CTA")
    folder_non = os.path.join(top_dict_pair, "non-contrast")
    clean_scan_folder(folder_cta, show=show)
    clean_scan_folder(folder_non, show=show)


def clean_multi_paired_folder(top_dict_pair, show=False):

    # requires pair with name strictly follow:
    # CTA1, non-contrast1; CTA2, non-contrast2, etc.

    assert os.listdir(top_dict_pair)

    remove_bad_files(top_dict_pair, show=show)

    scan_folder_name_list = os.listdir(top_dict_pair)

    if not len(scan_folder_name_list) >= 2 and len(scan_folder_name_list) % 2 == 0:
        print(top_dict_pair)
        print(scan_folder_name_list)
        assert False

    folder_cta = os.path.join(top_dict_pair, "CTA")
    folder_non = os.path.join(top_dict_pair, "non-contrast")

    if os.path.exists(folder_cta):
        assert len(scan_folder_name_list) == 2
        clean_scan_folder(folder_cta, show=show)
        clean_scan_folder(folder_non, show=show)
        return None
    else:
        assert len(scan_folder_name_list) > 2

    num_pair = int(len(scan_folder_name_list) / 2)

    for i in range(1, num_pair + 1):
        folder_cta = os.path.join(top_dict_pair, "CTA" + str(i))
        folder_non = os.path.join(top_dict_pair, "non-contrast" + str(i))
        clean_scan_folder(folder_cta, show=show)
        clean_scan_folder(folder_non, show=show)


def clean_paired_dataset(top_dict_dataset, multi_pair=False):
    """

    :param top_dict_dataset: all folders are paired scans
    :param multi_pair: True if there are many pair (pair name must in CTA1, non-contrast1, CTA2, non-contrast2, ...)
    :return:
    """
    remove_bad_files(top_dict_dataset, show=False)

    paired_dict_list = os.listdir(top_dict_dataset)[::-1]
    processed = 0

    for paired_dict in paired_dict_list:
        print("processing:", paired_dict, processed, '/', len(paired_dict_list))
        if not multi_pair:
            clean_paired_folder(os.path.join(top_dict_dataset, paired_dict), show=False)
        else:
            clean_multi_paired_folder(os.path.join(top_dict_dataset, paired_dict), show=False)
        processed += 1


if __name__ == '__main__':
    clean_paired_dataset('/Volumes/My Passport/paired_new_data_24-02-01/A436-A534/', multi_pair=True)
    exit()
    clean_paired_dataset('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209')
    exit()
    clean_paired_folder('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209/patient-id-2002 copy')
    exit()
    remove_bad_files('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209/patient-id-2002/CTA copy')
    clean_folder_structure('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209/patient-id-2002/CTA copy')
    exit()

    for path_ in file_operations.extract_all_file_path('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209/patient-id-2002/CTA copy'):
        print(path_)
    exit()
    # 2184 - 2194, 2209 empty files


