import os
import shutil
import time


def copyfile(source_path, save_path, show=True):
    """
    will overwrite is save path exist
    :param show
    :param source_path: path of the
    :param save_path:
    :return:
    """
    if show:
        print("copy path:", source_path)
        print("save_path:", save_path)
    shutil.copyfile(source_path, save_path)


def separate_path_to_file_structure(path):
    return list(path.split('/')[1:])


def merge_file_structure_to_path(list_name):
    return '/' + os.path.join(*list_name)


def get_father_dict(path_or_dict=None):
    if path_or_dict is None:
        return os.path.abspath(os.path.join(os.getcwd(), '..'))
    name_list = path_or_dict.split('/')
    valid_name_list = []
    for folder_name in name_list:
        if len(folder_name) == 0:
            continue
        valid_name_list.append(folder_name)

    new_path = '/'

    for folder_name in valid_name_list[:-1]:
        new_path = os.path.join(new_path, folder_name)
    return new_path


def copy_file_or_dir(source_path, save_path, show=True):
    """
    :param show
    :param source_path: path of the
    :param save_path:
    :return:
    """
    if show:
        print("copy path:", source_path)
        print("save_path:", save_path)

    if not os.path.isdir(source_path):
        father_dict = get_father_dict(save_path)
        if os.path.exists(father_dict):
            assert os.path.isdir(father_dict)
        else:
            os.makedirs(father_dict)
        shutil.copyfile(source_path, save_path)
    else:
        if os.path.exists(save_path):
            assert len(os.listdir(save_path)) == 0
            shutil.rmtree(save_path)
        shutil.copytree(source_path, save_path)


def move_file_or_dir(source_path, target_path, show=True):
    if show:
        print("move path from:", source_path)
        print("to target path:", target_path)
    shutil.move(source_path, target_path)


def extract_all_file_path(top_directory, end_with=None, start_with=None, name_contain=None):
    """

    :param top_directory:
    :param end_with: how the path end_with? like end with '.npy', then only extract path for ../file_name.npy
    :param start_with: file name start with, like if you want to remove '._' files
    :param name_contain: file name contain certain string
    :return: list of path
    """

    if end_with is not None:
        assert start_with is None and name_contain is None
    if start_with is not None:
        assert end_with is None and name_contain is None
    if name_contain is not None:
        assert start_with is None and end_with is None

    return_list = []
    if os.path.isfile(top_directory):
        if end_with is not None:
            if len(top_directory) <= len(end_with):
                return []
            else:
                if top_directory[-len(end_with):] == end_with:
                    return [top_directory, ]
                else:
                    return []
        elif start_with is not None:
            if len(top_directory) <= len(start_with):
                return []
            else:
                if top_directory.split('/')[-1][0: len(start_with)] == start_with:
                    return [top_directory, ]
                else:
                    return []
        elif name_contain is not None:
            if len(top_directory) <= len(name_contain):
                return []
            else:
                if name_contain in top_directory.split('/')[-1]:
                    return [top_directory, ]
                else:
                    return []
        else:
            return [top_directory, ]

    sub_dir_list = os.listdir(top_directory)
    for sub_dir in sub_dir_list:
        return_list = return_list + extract_all_file_path(os.path.join(top_directory, sub_dir),
                                                          end_with=end_with, start_with=start_with)

    return return_list


def extract_all_directory(top_directory, folder_name=None, folder_name_contain=None, complete_tree=True,
                          relative_dir=False):
    """

    :param top_directory:
    :param folder_name: what folder name you want? like 'CTA', then only extract directory like ./CTA
    :param folder_name_contain: what the name contain? like 'CTA', then only extract folder name contains 'CTA'
    :param complete_tree: True to return all directories, False to return directory not contain sub-directory
    (only have files)
    :param relative_dir: absolute dir = os.path.join(top_directory, relative_dir)
    :return: list of directory
    """

    if relative_dir:
        if not top_directory[-1] == '/':
            top_directory = top_directory + '/'
        new_return_list = []
        for absolute_dir in extract_all_directory(
                top_directory, folder_name, folder_name_contain, complete_tree, relative_dir=False):
            new_return_list.append(absolute_dir[len(top_directory):])
        return new_return_list

    if folder_name is not None:
        assert folder_name_contain is None

    return_list = []
    if os.path.isfile(top_directory):
        return []
    else:
        if folder_name is not None:
            current_folder_name = top_directory.split('/')[-1]
            if current_folder_name == folder_name:
                return_list.append(top_directory)
        elif folder_name_contain is not None:
            current_folder_name = top_directory.split('/')[-1]
            if folder_name_contain in current_folder_name:
                return_list.append(top_directory)
        else:
            return_list.append(top_directory)

    sub_dir_list = os.listdir(top_directory)

    if not complete_tree:
        check_contain_sub_directory = True
    else:
        check_contain_sub_directory = False
    for sub_dir in sub_dir_list:
        new_dir = os.path.join(top_directory, sub_dir)
        if check_contain_sub_directory:
            if os.path.isdir(new_dir):
                if top_directory in return_list:
                    return_list.remove(top_directory)
                check_contain_sub_directory = False
        return_list = return_list + extract_all_directory(new_dir, folder_name=folder_name,
                                                          folder_name_contain=folder_name_contain,
                                                          complete_tree=complete_tree, relative_dir=False)

    return return_list


def remove_path_or_directory(path_or_directory, show=True):
    if not os.path.exists(path_or_directory):
        if show:
            print("path or directory not exist:", path_or_directory)
            return
    if os.path.isfile(path_or_directory):
        os.remove(path_or_directory)
        if show:
            print("deleted path:", path_or_directory)
    else:
        shutil.rmtree(path_or_directory)
        if show:
            print("deleted directory:", path_or_directory)


def rename_file_or_folder_name(path_or_directory, new_name, show=True):
    assert type(new_name) is str
    new_path_or_directory = os.path.join(get_father_dict(path_or_directory), new_name)
    if new_path_or_directory == path_or_directory:
        return
    if show:
        print("old path:", path_or_directory)
        print("new_path:", new_path_or_directory)
    os.rename(path_or_directory, new_path_or_directory)


def get_ord_sum(string):
    ord_sum = 0
    for char in string:
        ord_sum += ord(char)
    return ord_sum


def split_sample_path_list_according_to_sor_sum(path_list, fold=(0, 1)):
    new_list = []
    for path in path_list:
        if get_ord_sum(path) % fold[1] == fold[0]:
            new_list.append(path)
    return new_list


def time_stamp_to_time(timestamp, show=False):
    time_struct = time.localtime(timestamp)
    time_new = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
    if show:
        print(time_new)
    return time_new


def show_file_times(path):
    print("File path:", path)
    print("Create time:", time_stamp_to_time(os.path.getctime(path)))
    print("Modify time:", time_stamp_to_time(os.path.getmtime(path)))
    print("Access time:", time_stamp_to_time(os.path.getatime(path)))


def remove_empty_folders(top_directory, remove_self=False, show=True):
    """

    :param top_directory:
    :param remove_self: if no files in top_directory, remove top_directory
    :param show: show directory when removing
    :return: None
    """
    walk = list(os.walk(top_directory))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            remove_path_or_directory(path, show=False)
            if show:
                print("remove empty folder:", path)

    if remove_self:
        if len(os.listdir(top_directory)) == 0:
            remove_path_or_directory(top_directory, show=False)
            if show:
                print("remove empty folder:", top_directory)


if __name__ == '__main__':

    path_list_bad = extract_all_file_path('/Volumes/My Passport/paired_new_data_24-01-12', end_with='.DS_Store')
    for path_ in path_list_bad:
        print("removing:", path_)
        os.remove(path_)

    exit()

    bad_path = extract_all_directory('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_High_Quality', folder_name_contain='CTPA')
    for path_ in bad_path:
        rename_file_or_folder_name(path_, 'CTA')
    exit()
    print(extract_all_directory('/home/zhoul0a/Downloads/CTPA', folder_name='dongyu', complete_tree=False))
