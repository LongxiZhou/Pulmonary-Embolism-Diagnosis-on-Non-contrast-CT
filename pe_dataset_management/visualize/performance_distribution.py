import Tool_Functions.Functions as Functions
import visualization.visualize_distribution.distribution_analysis as visualize
import os
from pe_dataset_management.basic_functions import find_patient_id_dataset_correspondence, get_all_scan_name
import numpy as np


def get_performance_list(dict_reports='/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
                                      'optimal/performance', fn_list=None):
    performance_list_original = []
    performance_list_registered = []

    if fn_list is None:
        fn_list = os.listdir(dict_reports)

    for fn in fn_list:
        if '.pickle' not in fn:
            fn = fn + '.pickle'

        report = Functions.pickle_load_object(os.path.join(dict_reports, fn))
        if 'Z194' in fn:
            continue

        if report['guide mask dice on 256 original'] > 0.95:
            continue

        performance_list_original.append(report['guide mask dice on 256 original'])
        performance_list_registered.append(report['guide mask dice on 256 registered'])

    visualize.histogram_list(
        performance_list_original, interval=50, save_path='/data_disk/pulmonary_embolism/temp_files/before.svg',
        range_show=(0, 1))
    visualize.histogram_list(
        performance_list_registered, interval=50, save_path='/data_disk/pulmonary_embolism/temp_files/after.svg',
        range_show=(0, 1))

    print(len(performance_list_original))
    print(np.average(performance_list_original), np.std(performance_list_original))
    print(np.average(performance_list_registered), np.std(performance_list_registered))

    return performance_list_original, performance_list_registered


def visualize_registration_effect(scan_name, top_dict='/data_disk/CTA-CT_paired-dataset'):
    """

    :param scan_name: None for visualize all scans
    :param top_dict:
    :return:
    """
    # three view
    if scan_name is not None:
        if len(scan_name) <= 4:
            scan_name = scan_name + '.npz'
        if len(scan_name) > 4:
            if not scan_name[-4:] == '.npz':
                scan_name = scan_name + '.npz'
    dataset_dict_cta, dataset_dict_non_contrast = \
        find_patient_id_dataset_correspondence(top_dict=top_dict, scan_name=scan_name, check_pair=False)[scan_name[:-4]]

    print(dataset_dict_cta, dataset_dict_non_contrast)


if __name__ == '__main__':

    fn_good_register = []
    for name in os.listdir('/data_disk/CTA-CT_paired-dataset/temp_files/no_need_translate'):
        fn_good_register.append(name[:-6])

    get_performance_list('/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/'
                         'optimal/performance', fn_list=fn_good_register)
    get_performance_list(fn_list=fn_good_register)
    exit()
    visualize_registration_effect('Z154')
    exit()
    get_performance_list()
    exit()
    visualize_registration_effect('Z154')
