import pe_dataset_management.basic_functions as basic_functions
import os


def get_registration_quality():
    dir_very_good_quality = '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/' \
                            'visualization_optimal/manual_classify_quality/very_good_quality'
    dir_good_quality = '/data_disk/CTA-CT_paired-dataset/registration_from_cta_to_non_contrast/' \
                       'visualization_optimal/manual_classify_quality/good_quality'

    name_list_very_good_registration = []
    name_list_good_registration = []

    for fn in os.listdir(dir_good_quality):
        name_list_good_registration.append(fn[:-6])

    for fn in os.listdir(dir_very_good_quality):
        name_list_very_good_registration.append(fn[:-6])

    name_set_very_good_registration = set(name_list_very_good_registration)
    name_set_good_registration = set(name_list_good_registration)

    return name_set_good_registration, name_set_very_good_registration


def get_pe_pair_quality():
    scan_name_list_pe_high_quality = basic_functions.get_all_scan_name(scan_class='PE', dir_key_word='PE_High_Quality')
    scan_name_list_pe_low_quality = basic_functions.get_all_scan_name(
        scan_class='PE', dir_key_word='PE_Low_Quality', dir_exclusion_key_word='CT-after-CTA',
        fn_exclusion_key_word='zryh')

    manual_checked_high_quality = \
        ['patient-id-017', 'patient-id-021', 'patient-id-122', 'patient-id-171', 'patient-id-514',
         'patient-id-269']
    manual_checked_low_quality = \
        ['patient-id-023', 'patient-id-054', 'patient-id-064', 'patient-id-074', 'patient-id-050',
         'patient-id-554']

    scan_name_set_pe_low_quality = set(scan_name_list_pe_low_quality) | set(manual_checked_low_quality)
    scan_name_set_pe_high_quality = set(scan_name_list_pe_high_quality) | set(manual_checked_high_quality)

    scan_name_set_pe_low_quality = \
        scan_name_set_pe_low_quality - (scan_name_set_pe_low_quality & scan_name_set_pe_high_quality)

    return scan_name_set_pe_low_quality, scan_name_set_pe_high_quality


def get_quality_of_scan_name(use_existing=True):

    if use_existing:
        from Tool_Functions.Functions import pickle_load_object
        fn_good_pair_good_registration = pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_good_pair_good_registration.pickle')
        fn_good_pair_excellent_registration = pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_good_pair_excellent_registration.pickle')
        fn_excellent_pair_good_registration = pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_excellent_pair_good_registration.pickle')
        fn_excellent_pair_excellent_registration = pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/'
            'fn_list-PE_excellent_pair_excellent_registration.pickle')

    else:
        scan_name_set_pe_low_quality, scan_name_set_pe_high_quality = get_pe_pair_quality()
        name_set_good_registration, name_set_very_good_registration = get_registration_quality()

        fn_excellent_pair_excellent_registration = scan_name_set_pe_high_quality & name_set_very_good_registration
        fn_excellent_pair_good_registration = scan_name_set_pe_high_quality & name_set_good_registration
        fn_good_pair_excellent_registration = scan_name_set_pe_low_quality & name_set_very_good_registration
        fn_good_pair_good_registration = scan_name_set_pe_low_quality & name_set_good_registration

    return fn_good_pair_good_registration, fn_good_pair_excellent_registration, \
        fn_excellent_pair_good_registration, fn_excellent_pair_excellent_registration


if __name__ == '__main__':
    with_annotation = set()
    for name_set in get_quality_of_scan_name():
        with_annotation = with_annotation | name_set
    print(len(with_annotation))
    exit()
    a, b, c, d = get_quality_of_scan_name(True)

    print(len(a), len(b), len(c), len(d))
    print(len(a) + len(b) + len(c) + len(d))
    print(len(a & b), len(a & c), len(a & d), len(b & c), len(b & d), len(c & d))

    exit()
    import Tool_Functions.Functions as Functions
    Functions.pickle_save_object(
        '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_good_pair_good_registration.pickle', a)
    Functions.pickle_save_object(
        '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_good_pair_excellent_registration.pickle', b)
    Functions.pickle_save_object(
        '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_excellent_pair_good_registration.pickle', c)
    Functions.pickle_save_object(
        '/data_disk/pulmonary_embolism_final/pickle_objects/fn_list-PE_excellent_pair_excellent_registration.pickle', d)
    exit()
