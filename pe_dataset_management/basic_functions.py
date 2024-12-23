import os
import Tool_Functions.Functions as Functions


def get_dataset_relative_path(scan_class='All'):
    if scan_class == 'PE':
        name_list_modality = ['PE_High_Quality', 'PE_Low_Quality', os.path.join('may_not_pair', 'PE'),
                              os.path.join('strange_data', 'PE')]
    elif scan_class == 'Normal':
        name_list_modality = ['Normal_High_Quality', 'Normal_Low_Quality', os.path.join('may_not_pair', 'Normal'),
                              os.path.join('strange_data', 'Normal')]
    elif scan_class == 'Temp':
        name_list_modality = ['Temp_High_Quality', 'Temp_Low_Quality', os.path.join('may_not_pair', 'Temp'),
                              os.path.join('strange_data', 'Temp')]
    else:
        assert scan_class == 'All'
        return get_dataset_relative_path('PE') + get_dataset_relative_path('Normal') + get_dataset_relative_path('Temp')

    name_list_sub_dataset_low_quality = ['long_CTA-CT_interval', 'CT-after-CTA', 'good_CTA-CT_interval_but_bad_dcm',
                                         'CTA > 2 days after CT']
    name_list_dataset = []
    for modality in name_list_modality:
        if 'Low' in modality:
            for sub_dataset in name_list_sub_dataset_low_quality:
                name_list_dataset.append(os.path.join(modality, sub_dataset))
        else:
            name_list_dataset.append(modality)

    return name_list_dataset


def find_patient_id_dataset_correspondence(scan_name=None, top_dict='/data_disk/CTA-CT_paired-dataset',
                                           check_pair=False, strip=False):
    """

    :param strip:
    :param scan_name: 'patient-id-21340562'
    :param top_dict:
    :param check_pair
    :return: {patient-id: [data_dict_CTA, data_dict_non_contrast]}
            e.g., {"patient-id-21340562":, ['/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality,
                                            '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality']}
    """
    if scan_name is not None:
        if len(scan_name) <= 4:
            scan_name = scan_name + '.npz'
        if len(scan_name) > 4:
            if not scan_name[-4:] == '.npz':
                scan_name = scan_name + '.npz'

    if check_pair:
        check_ct_non_contrast_pair(top_dict)

    top_dict_cta = os.path.join(top_dict, 'dataset_CTA')
    top_dict_non_contrast = os.path.join(top_dict, 'dataset_non_contrast')

    relative_dataset_path_list = get_dataset_relative_path()
    return_dict = {}
    for relative_dataset in relative_dataset_path_list:
        dataset_dict_cta = os.path.join(top_dict_cta, relative_dataset)
        dataset_dict_non_contrast = os.path.join(top_dict_non_contrast, relative_dataset)
        dict_rescaled_cta = os.path.join(top_dict_cta, relative_dataset, 'rescaled_ct')
        dict_rescaled_non_contrast = os.path.join(top_dict_non_contrast, relative_dataset, 'rescaled_ct')
        if not os.path.exists(dict_rescaled_cta) or not os.path.exists(dict_rescaled_non_contrast):
            continue
        file_name_list = os.listdir(dict_rescaled_cta)
        if scan_name is not None:
            if scan_name not in file_name_list:
                continue

        if scan_name is None:
            for file_name in file_name_list:
                path_non_contrast = os.path.join(dict_rescaled_non_contrast, file_name)
                if not os.path.exists(path_non_contrast):
                    print(file_name, "exist in CTA but not exist in non-contrast CT dataset path:", path_non_contrast)
                    continue

                return_dict[Functions.strip_suffix(file_name)] = [dataset_dict_cta, dataset_dict_non_contrast]
        else:
            return_dict[Functions.strip_suffix(scan_name)] = [dataset_dict_cta, dataset_dict_non_contrast]

    if len(return_dict) == 0:
        if scan_name is not None:
            raise ValueError("scan name not found for", scan_name)
        raise ValueError("no paired scan in the database")

    if strip:
        assert len(return_dict) == 1
        return return_dict[list(return_dict.keys())[0]]

    return return_dict


def find_original_dcm_folders(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset'):
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    top_dict_dcm_files = '/data_disk/CTA-CT_paired-dataset/paired_dcm_files'

    dict_cta = os.path.join(top_dict, 'dataset_CTA') + '/'

    scan_name = Functions.strip_suffix(scan_name)

    data_dict_cta, data_dict_non_contrast = find_patient_id_dataset_correspondence(
        scan_name=scan_name, top_dict=top_dict)[scan_name]

    dict_dcm_cta = os.path.join(top_dict_dcm_files, data_dict_cta[len(dict_cta)::], scan_name, 'CTA')
    dict_dcm_non_contrast = os.path.join(top_dict_dcm_files, data_dict_cta[len(dict_cta)::], scan_name, 'non-contrast')
    return dict_dcm_cta, dict_dcm_non_contrast


def get_all_scan_name(top_dict='/data_disk/CTA-CT_paired-dataset', scan_class='All', dir_key_word=None,
                      dir_exclusion_key_word=None, fn_key_word=None, fn_exclusion_key_word=None):
    # scan_class in ['PE', 'Normal', 'All']
    relative_dataset_path_list = get_dataset_relative_path(scan_class=scan_class)
    name_list_ct = []
    for relative_dataset in relative_dataset_path_list:
        if dir_key_word is not None:
            if type(dir_key_word) is str:
                if dir_key_word not in relative_dataset:
                    continue
            elif type(dir_key_word) is list:
                skip = True
                for key in dir_key_word:
                    if key in relative_dataset:
                        skip = False
                if skip:
                    continue
            else:
                raise ValueError("key word should be str or list")

        if dir_exclusion_key_word is not None:
            if type(dir_exclusion_key_word) is str:
                if dir_exclusion_key_word in relative_dataset:
                    continue
            elif type(dir_exclusion_key_word) is list:
                skip = False
                for exclusion_key in dir_exclusion_key_word:
                    if exclusion_key in relative_dataset:
                        skip = True
                if skip:
                    continue
            else:
                raise ValueError("exclusion key word should be str or list")

        dict_rescaled_ct = os.path.join(top_dict, 'dataset_CTA', relative_dataset, 'rescaled_ct')
        if not os.path.exists(dict_rescaled_ct):
            continue
        file_name_list = os.listdir(dict_rescaled_ct)
        for file_name in file_name_list:
            if fn_key_word is not None:
                if type(fn_key_word) is str:
                    if fn_key_word not in file_name:
                        continue
                elif type(fn_key_word) is list:
                    skip = True
                    for key in fn_key_word:
                        if key in file_name:
                            skip = False
                    if skip:
                        continue
                else:
                    raise ValueError("key word should be str or list")
            if fn_exclusion_key_word is not None:
                if type(fn_exclusion_key_word) is str:
                    if fn_exclusion_key_word in file_name:
                        continue
                elif type(fn_exclusion_key_word) is list:
                    skip = False
                    for exclusion_key in fn_exclusion_key_word:
                        if exclusion_key in file_name:
                            skip = True
                    if skip:
                        continue
                else:
                    raise ValueError("exclusion key word should be str or list")
            name_list_ct.append(Functions.strip_suffix(file_name))
    return name_list_ct


def check_ct_non_contrast_pair(top_dict='/data_disk/CTA-CT_paired-dataset'):
    top_dict_cta = os.path.join(top_dict, 'dataset_CTA')
    top_dict_non_contrast = os.path.join(top_dict, 'dataset_non_contrast')

    name_list_cta = get_all_scan_name(top_dict_cta)
    assert len(name_list_cta) == len(set(name_list_cta))

    name_list_non_contrast = get_all_scan_name(top_dict_non_contrast)
    assert len(name_list_non_contrast) == len(set(name_list_non_contrast))


def get_file_name_do_not_have_clot_gt(top_dict='/data_disk/CTA-CT_paired-dataset', scan_class='PE'):
    # return a list like ['scan-name.npz', ]

    name_list_with_gt = []
    relative_dataset_path_list = get_dataset_relative_path(scan_class=scan_class)
    for relative_dataset in relative_dataset_path_list:

        dict_rescaled_ct = os.path.join(top_dict, 'dataset_CTA', relative_dataset, 'rescaled_ct')
        if not os.path.exists(dict_rescaled_ct):
            continue

        clot_gt_dict = os.path.join(top_dict, 'dataset_CTA', relative_dataset, 'clot_gt')
        if os.path.exists(clot_gt_dict):
            fn_with_gt_sub_list = []
            for fn in os.listdir(clot_gt_dict):
                fn_with_gt_sub_list.append(Functions.strip_suffix(fn))
            name_list_with_gt = name_list_with_gt + fn_with_gt_sub_list

    name_set_all = set(get_all_scan_name(top_dict, scan_class=scan_class))

    return_list = list(name_set_all.difference(set(name_list_with_gt)))
    return_list.sort()
    return return_list


# def delete_features


if __name__ == '__main__':

    print(find_patient_id_dataset_correspondence('NJ0030'))

    print(len(get_all_scan_name()))
    print(get_all_scan_name()[:10])
    exit()
