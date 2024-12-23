import os
import pe_dataset_management.ct_cta_paired_dataset.establish_rescaled_ct as func_check_time
import Tool_Functions.Functions as Functions
import pydicom


def check_prior_and_pair(top_dict_folder, pickle_save_path=None):
    list_ct_time_prior = []

    fn_list = os.listdir(top_dict_folder)
    fn_list.sort()

    def get_ct_time_prior(cta_dir, non_dir):
        print("CTA folder:", cta_dir)
        print("Non-contrast folder:", non_dir)
        time_prior = func_check_time.check_scan_time(non_dir, cta_dir, return_time_prior_only=True)
        result_scan_pair = func_check_time.check_pair(non_dir, cta_dir)
        print('\n', "From same patient:", result_scan_pair, "  ct prior time", time_prior, '\n')
        return time_prior

    scan_passed = 0

    for fn in fn_list:
        print("processing:", fn, scan_passed, '/', len(fn_list))
        patient_folder = os.path.join(top_dict_folder, fn)

        num_pair = int(len(os.listdir(patient_folder)) / 2)

        if num_pair == 1:
            cta_folder = os.path.join(patient_folder, 'CTA')
            non_contrast_folder = os.path.join(patient_folder, 'non-contrast')
            if not os.path.exists(cta_folder):
                cta_folder = os.path.join(patient_folder, 'CTA1')
                non_contrast_folder = os.path.join(patient_folder, 'non-contrast1')
            ct_time_prior = get_ct_time_prior(cta_folder, non_contrast_folder)
            list_ct_time_prior.append(ct_time_prior)
        else:
            for pair_count in range(1, num_pair + 1):
                cta_folder = os.path.join(patient_folder, 'CTA' + str(pair_count))
                non_contrast_folder = os.path.join(patient_folder, 'non-contrast' + str(pair_count))
                ct_time_prior = get_ct_time_prior(cta_folder, non_contrast_folder)
                list_ct_time_prior.append(ct_time_prior)

        scan_passed += 1

    if pickle_save_path is None:
        return
    Functions.pickle_save_object(pickle_save_path,
                                 list_ct_time_prior, buffer_path='/Users/richard/Desktop/transfer/temp')


def print_data_source_hospital(top_dict_folder):

    def get_source_from_dcm_path(dcm_path):
        tag_source = pydicom.dataset.Tag('InstitutionName')
        dcm_file = pydicom.read_file(dcm_path)
        if tag_source in list(dcm_file.keys()):
            return dcm_file[tag_source].value
        else:
            return None

    fn_list = os.listdir(top_dict_folder)
    fn_list.sort()

    scan_passed = 0

    for fn in fn_list:
        print("processing:", fn, scan_passed, '/', len(fn_list))
        patient_folder = os.path.join(top_dict_folder, fn)
        cta_folder = os.path.join(patient_folder, 'CTA')
        if not os.path.exists(cta_folder):
            cta_folder = os.path.join(patient_folder, 'CTA1')
        non_contrast_folder = os.path.join(patient_folder, 'non-contrast')
        if not os.path.exists(non_contrast_folder):
            non_contrast_folder = os.path.join(patient_folder, 'non-contrast1')

        cta_dcm_fn_list = os.listdir(cta_folder)
        non_dcm_fn_list = os.listdir(non_contrast_folder)

        path_cta_dcm = os.path.join(cta_folder, cta_dcm_fn_list[0])
        path_non_dcm = os.path.join(non_contrast_folder, non_dcm_fn_list[0])

        print("CTA from         ", get_source_from_dcm_path(path_cta_dcm))
        print("Non-contrast from", get_source_from_dcm_path(path_non_dcm))
        print("\n")

        scan_passed += 1


if __name__ == '__main__':
    check_prior_and_pair('/Volumes/My Passport/paired_new_data_24-02-01/A436-A534/',
                         '/Volumes/My Passport/paired_new_data_24-02-01/list_ct_time_prior.pickle')
    exit()
    print_data_source_hospital('/Volumes/My Passport/paired_new_data_24-02-01/A436-A534/')

    exit()
    check_prior_and_pair('/Volumes/My Passport/paired_new_data_24-01-12/patient-id-2001-2209/',
                         '/Volumes/My Passport/paired_new_data_24-01-12/list_ct_time_prior.pickle')
