

def find_strings_in_a_file(path, list_strings, report_not_found=True):
    """

    :param path:
    :param list_strings: string or list of strings ['', ]
    :param report_not_found:
    :return:
    """
    if type(list_strings) is str:
        list_strings = [list_strings, ]
    found = False
    with open(path, 'r') as f:
        for index, line in enumerate(f):
            # search string
            for target_string in list_strings:
                if target_string in line:
                    print("line", index + 1, "found", target_string, 'in', path)
                    found = True
    if report_not_found and not found:
        print(list_strings, 'does not exist in', path)
    return found


def find_string_in_a_directory(directory_path, list_strings, end_with='.py', report_not_found=True):
    """

    :param directory_path:
    :param list_strings: string or list of strings ['', ]
    :param end_with:
    :param report_not_found:
    :return:
    """
    from Tool_Functions.file_operations import extract_all_file_path

    path_list = extract_all_file_path(directory_path, end_with, start_with=None, name_contain=None)

    found = False
    for path in path_list:
        if find_strings_in_a_file(path, list_strings, report_not_found=False):
            found = True

    if report_not_found and not found:
        print(list_strings, 'does not exist in', directory_path)
    return found


if __name__ == '__main__':
    find_string_in_a_directory('/home/zhoul0a/Desktop/Longxi_Platform/', 'samples_for_performance_evaluation_cta_confirm')



