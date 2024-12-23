import os


def apply_analysis_database(dict_for_dataset_structure, list_top_dict_source, top_dict_save, list_func_load_item,
                            func_analysis, func_save, func_check_processed):
    """
    database is understood as several dataset
    dataset is understood as "ct_of_same_property", go into the dict_for_dataset_structure, must give and only give a
    list of dataset directory, each contains multiple "ct_of_same_property". Such dataset directory will be used to
    load item from top_dict_rescaled_ct, and will be used for saving analysis results

    "func_analysis" input len(list_top_dict_source) number of items, and forms a new analysis results

    :param dict_for_dataset_structure
    :param list_top_dict_source: a list of directory, each directory must be in same structure, only the file item are
     different: same_name plus file appendix
    :param top_dict_save: analysis result will be save with func_save(to_dict_save, analysis_results)
    :param list_func_load_item: item will be loaded by the functions.
    :param func_analysis: the function to get the analysis results
    :param func_save: the function to save the analysis results
    :param func_check_processed
    :return: None
    """
    assert len(list_func_load_item) == len(list_top_dict_source)

    # in each directory is a number of ct scans
    list_dataset_top_directory = extract_dataset_structure(dict_for_dataset_structure)
    processed = 0
    for dataset in list_dataset_top_directory:
        print("############################################################")
        print("processing dataset at:", os.path.join(dict_for_dataset_structure, dataset))
        print("dataset left:", len(list_dataset_top_directory) - processed)
        print("############################################################")

        scan_name_list = os.listdir(os.path.join(dict_for_dataset_structure, dataset))
        # scan_name in .npy

        dict_save = os.path.join(top_dict_save, dataset)
        list_dict_source = []
        for top_dict_source in list_top_dict_source:
            list_dict_source.append(os.path.join(top_dict_source, dataset))

        apply_analysis_dataset(scan_name_list, list_dict_source, dict_save, list_func_load_item, func_analysis,
                               func_save, func_check_processed)
        processed += 1


def apply_analysis_dataset(scan_name_list, list_dict_source, dict_save, list_func_load_item, func_analysis, func_save,
                           func_check_processed):
    """
    dataset is understood as "ct_of_same_property", go into the dict_for_dataset_structure, must give and only give a
    list of dataset directory, each contains multiple "ct_of_same_property".
    :param scan_name_list:
    :param list_dict_source:
    :param dict_save:
    :param list_func_load_item:  each func in func(dict_source, scan_name), returns the item for the scan_name
    :param func_analysis:
    :param func_save:
    :param func_check_processed
    :return: None
    """
    assert len(list_func_load_item) == len(list_dict_source)

    count = 0

    for scan_name in scan_name_list:
        # scan_name in .npy

        print("processing:", scan_name, len(scan_name_list) - count, 'left')
        if func_check_processed(dict_save, scan_name):
            print("processed")
            count += 1
            continue

        input_item_list = []
        for source_id in range(len(list_dict_source)):
            input_item = list_func_load_item[source_id](list_dict_source[source_id], scan_name)
            input_item_list.append(input_item)

        results = func_analysis(input_item_list)

        if results is None:
            count += 1
            print("cannot process")
            continue

        func_save(dict_save, scan_name, results)

        count += 1


def extract_dataset_structure(dict_for_dataset_structure, show=False):
    if not dict_for_dataset_structure[-1] == '/':
        dict_for_dataset_structure = dict_for_dataset_structure + '/'
    list_dataset_top_directory = []  # in each directory is a number of ct scans
    for root, dirs, files in os.walk(dict_for_dataset_structure):
        if show:
            print(root, dirs, files)
            print(len(dirs), len(files))
        if len(dirs) == 0 and len(files) > 0:
            list_dataset_top_directory.append(root[len(dict_for_dataset_structure)::])
        if len(dirs) > 0 and len(files) > 0:
            print("wrong directory and file name:", dirs, files)
            raise ValueError("directory and file should not occur simultaneously")
    if show:
        print("extracted", len(list_dataset_top_directory), 'dataset:')
        print(list_dataset_top_directory)
    return list_dataset_top_directory


def pipeline_get_depth_and_center_line():
    import Tool_Functions.Functions as Functions
    import analysis.center_line_and_depth_3D as center_line_and_depth
    import numpy as np
    dict_for_dataset_structure = '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct/'
    list_top_dict_source = ['/media/zhoul0a/New Volume/rescaled_ct_and_semantics/semantics/']
    top_dict_save = '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/depth_and_center-line/'

    def load_vessel_mask(dict_dataset, scan_name):  # func_load
        print(os.path.join(dict_dataset, 'blood_mask/', scan_name[:-4] + '.npz'))
        if not os.path.exists(os.path.join(dict_dataset, 'blood_mask/', scan_name[:-4] + '.npz')):
            print("file not exist")
            return None
        return np.load(os.path.join(dict_dataset, 'blood_mask/', scan_name[:-4] + '.npz'))['array']

    list_func_load_item = [load_vessel_mask]

    def get_center_line_and_depth(input_list):  # func_analysis
        for input_item in input_list:
            if input_item is None:
                return None
        blood_vessel_mask = input_list[0]
        depth_mask = center_line_and_depth.get_surface_distance(blood_vessel_mask, strict=True)
        center_line_mask = center_line_and_depth.get_center_line(blood_vessel_mask,
                                                                 surface_distance=depth_mask, search_radius=4)
        return depth_mask, center_line_mask

    def save_function(dict_save, scan_name, item_list):
        depth_mask, center_line_mask = item_list
        Functions.save_np_array(os.path.join(dict_save, 'depth_array/'), scan_name[:-4] + '.npz', depth_mask,
                                compress=True)
        Functions.save_np_array(os.path.join(dict_save, 'center_line_mask/'), scan_name[:-4] + '.npz', center_line_mask,
                                compress=True)

    def func_check_processed(dict_save, scan_name):
        if not os.path.exists(os.path.join(dict_save, 'center_line_mask/', scan_name[:-4] + '.npz')):
            return False
        if not os.path.exists(os.path.join(dict_save, 'center_line_mask/', scan_name[:-4] + '.npz')):
            return False
        return True

    apply_analysis_database(dict_for_dataset_structure, list_top_dict_source, top_dict_save, list_func_load_item,
                            get_center_line_and_depth, save_function, func_check_processed)


if __name__ == '__main__':
    pipeline_get_depth_and_center_line()
    exit()
    print(extract_dataset_structure('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct/'))
    print(extract_dataset_structure('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/semantics/'))
