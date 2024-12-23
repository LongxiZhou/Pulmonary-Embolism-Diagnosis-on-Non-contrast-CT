import os
import Tool_Functions.Functions as Functions
import Tool_Functions.file_operations as file_operations
import pydicom
from datetime import date
from format_convert.dcm_np_converter_new import get_resolution_from_dcm
import sys

sys.path.append('/home/zhoul0a/Desktop/longxi_platform/')


def shuttle_dcm_pair(top_dict_pair, dataset_type='PE',
                     target_dict_database='/data_disk/CTA-CT_paired-dataset/paired_dcm_files',
                     class_list=("High_Quality", "CT-after-CTA", "long_CTA-CT_interval",
                                 "good_CTA-CT_interval_but_bad_dcm", "CTA > 2 days after CT", "may_not_pair"),
                     exclusion_class_list=None, clean_data=True, fold=(0, 1)):
    """

    :param fold:
    :param clean_data:
    :param top_dict_pair:
    :param dataset_type:
    :param target_dict_database:
    :param class_list: classes that the program will automatically classify pairs
    :param exclusion_class_list: a list, which class we need to manual check, like ("may_not_pair")
    :return: None
    """
    patient_id_list = os.listdir(top_dict_pair)
    patient_id_list = Functions.split_list_by_ord_sum(patient_id_list, fold=fold)

    num_patients = len(patient_id_list)

    class_list = list(class_list)
    if exclusion_class_list is not None:
        exclusion_class_list = list(exclusion_class_list)
        for exclusion_class in exclusion_class_list:
            class_list.remove(exclusion_class)

    processed_count = 0
    for patient_id in patient_id_list:
        print('\n\n', "########################\n", "processing", patient_id, num_patients - processed_count, 'left')

        top_dict_case = os.path.join(top_dict_pair, patient_id)
        content_name_list = os.listdir(top_dict_case)

        # step 1) check sub-dir in each case contains non-contrast and contrast
        check_content_name(content_name_list)

        dict_non = os.path.join(top_dict_case, 'non-contrast')
        dict_cta = os.path.join(top_dict_case, 'CTA')
        if clean_data:
            clean_dataset(dict_cta)
            clean_dataset(dict_non)

        # step 2) whether CTA and non-contrast from the same patient
        if not check_pair(dict_non, dict_cta):
            if "may_not_pair" in class_list:
                target_dict = os.path.join(target_dict_database, "may_not_pair", dataset_type, patient_id)
                print("moving files...", patient_id)
                file_operations.move_file_or_dir(top_dict_case, target_dict)

            print(patient_id, "is classified as: may_not_pair")
            processed_count += 1

            continue

        # step 3) check_time
        good_relative_time, class_name = check_scan_time(dict_non, dict_cta, show=True)

        # step 4) check_resolution
        good_resolution = check_resolution(dict_non, dict_cta)

        # step 5) determine target_dict

        if good_relative_time and good_resolution:
            target_dict = os.path.join(target_dict_database, dataset_type + "_High_Quality", patient_id)
            print("moving files...", patient_id)
            file_operations.move_file_or_dir(top_dict_case, target_dict)
            print(patient_id, "is classified as:", dataset_type + "_High_Quality")
            processed_count += 1
            continue

        if good_relative_time and not good_resolution:
            print(patient_id, "is classified as:", dataset_type + "_Low_Quality/" + "good_CTA-CT_interval_but_bad_dcm")
            if "good_CTA-CT_interval_but_bad_dcm" in class_list:
                target_dict = os.path.join(
                    target_dict_database, dataset_type + "_Low_Quality", "good_CTA-CT_interval_but_bad_dcm", patient_id)
                print("moving files...", patient_id)
                file_operations.move_file_or_dir(top_dict_case, target_dict)
            processed_count += 1
            continue

        if not good_relative_time:
            print(patient_id, "is classified as:",  dataset_type + "_Low_Quality/" + class_name)
            if class_name in class_list:
                target_dict = os.path.join(target_dict_database, dataset_type + "_Low_Quality", class_name, patient_id)
                print("moving files...", patient_id)
                file_operations.move_file_or_dir(top_dict_case, target_dict, show=True)
            processed_count += 1


def clean_dataset(top_dict, func_clean=None):
    import Tool_Functions.file_operations as file_operations
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


def check_files_1_to_2(top_dict_patient, show=True, whether_check_time=True, whether_check_pair=True,
                       semantic=None):
    """
    Step 1 to 2
    1) each patient should has
    a folder: 'non-contrast', a folder: 'CTA', a file 'annotation_for_blood_clot_on_CTA.mha' (optional)
    2) the scan time from 'non-contrast' and 'CTA' should be very close for PE patient
    :param semantic: if not None, will automatically move low quality file to relevant place
    :param whether_check_pair:
    :param whether_check_time: only for High Quality pairs
    :param show:
    :param top_dict_patient:
    :return: None
    """
    print("########################################")
    print("# Checking Scan Time and Patient Pair...")
    print("########################################\n\n\n")

    if semantic is not None:
        assert semantic in ["Normal", "PE", "Temp"]

    patient_id_list = os.listdir(top_dict_patient)
    patient_id_list.sort()

    num_patients = len(patient_id_list)

    good_list = ['patient-id-135']

    processed_count = 0
    for patient_id in patient_id_list:
        print(patient_id, num_patients - processed_count, 'left')

        top_dict_content = os.path.join(top_dict_patient, patient_id)

        content_name_list = os.listdir(top_dict_content)

        # step 1)
        check_content_name(content_name_list)

        # step 2)
        if patient_id not in good_list:
            if whether_check_pair:
                if not check_pair(os.path.join(top_dict_content, 'non-contrast'),
                                  os.path.join(top_dict_content, 'CTA')):
                    if semantic is None:
                        raise ValueError("may not pair")
                    else:
                        assert "High_Quality" in top_dict_content
                        target_dict = Functions.get_father_dict(top_dict_patient)
                        assert "High_Quality" not in target_dict
                        target_dict = os.path.join(target_dict, 'may_not_pair', semantic, patient_id)
                        file_operations.move_file_or_dir(top_dict_content, target_dict, show=True)
                        continue

            if whether_check_time:
                status, reason = check_scan_time(os.path.join(top_dict_content, 'non-contrast'),
                                                 os.path.join(top_dict_content, 'CTA'), show=show)
                if not status:
                    if semantic is None:
                        raise ValueError("not good time")
                    assert "High_Quality" in top_dict_content
                    target_dict = Functions.get_father_dict(top_dict_patient)
                    # /data_disk/CTA-CT_paired-dataset/paired_dcm_files
                    assert "High_Quality" not in target_dict

                    target_dict = os.path.join(target_dict, semantic + "_Low_Quality")

                    if reason == "CTA before CT":
                        target_dict = os.path.join(target_dict, "CT-after-CTA")
                    if reason == "CTA CT date not match":
                        raise ValueError("not good time")
                    if reason == "CTA long after CT":
                        target_dict = os.path.join(target_dict, "long_CTA-CT_interval")

                    target_dict = os.path.join(target_dict, patient_id)

                    file_operations.move_file_or_dir(top_dict_content, target_dict, show=True)

        if show:
            print("\n")
        processed_count += 1


def check_content_name(content_name_list):
    assert 'CTA' in content_name_list
    assert 'non-contrast' in content_name_list
    if len(content_name_list) == 3:
        assert 'annotation_for_blood_clot_on_CTA.mha' in content_name_list
    assert len(content_name_list) < 4


def get_study_date_time(dcm_dict):
    dcm_name_list = os.listdir(dcm_dict)
    dcm_path = os.path.join(dcm_dict, dcm_name_list[0])
    dcm_example = pydicom.read_file(dcm_path)

    # check the date
    if "AcquisitionDateTime" in list(dcm_example.keys()):
        date_time = dcm_example["AcquisitionDateTime"].value
        study_date = date_time[0:8]  # type is str, like '20200210'
        study_time = date_time[8:14]  # type is str, like '093209'

    else:
        study_date = ''
        date_key_list = ["StudyDate", "AcquisitionDate", "SeriesDate"]
        for date_key in date_key_list:
            if date_key in list(dcm_example.keys()):
                study_date = dcm_example[date_key].value
                if len(study_date) > 0:
                    break

        study_time = ''
        time_key_list = ["StudyTime", "AcquisitionTime", "SeriesTime", ]
        for time_key in time_key_list:
            if time_key in list(dcm_example.keys()):
                study_time = dcm_example[time_key].value
                if len(study_time) > 0:
                    break

    if not len(study_date) > 0 or not len(study_time) > 0:
        print("wrong names:")
        print(dcm_path)
        print(dcm_example)
        exit()
        assert len(study_date) > 0
        assert len(study_time) > 0

    return study_date, study_time


def check_scan_time(dict_ct, dict_cta, show=True, return_time_prior_only=False):

    study_date_ct, study_time_ct = get_study_date_time(dict_ct)
    study_date_cta, study_time_cta = get_study_date_time(dict_cta)

    def get_min_prior(str_a, str_b):
        """
        minutes str_a is prior to str_b
        :param str_a: like '093209'
        :param str_b: like '093539'
        :return: float, minutes str_a is prior to str_b, like 3.5
        """
        prior = (int(str_b[0:2]) - int(str_a[0:2])) * 60 + int(str_b[2:4]) - int(str_a[2:4]) + \
                (int(str_b[4:6]) - int(str_a[4:6])) / 60

        return prior

    min_ct_prior_to_cta = get_min_prior(study_time_ct, study_time_cta)
    date_ct = date(int(study_date_ct[0:4]), int(study_date_ct[4:6]), int(study_date_ct[6:8]))
    date_cta = date(int(study_date_cta[0:4]), int(study_date_cta[4:6]), int(study_date_cta[6:8]))
    date_prior = date_cta - date_ct
    date_prior = date_prior.days
    min_ct_prior_to_cta = min_ct_prior_to_cta + date_prior * 24 * 60

    if return_time_prior_only:
        return min_ct_prior_to_cta

    if 0 <= min_ct_prior_to_cta < 120:

        if min_ct_prior_to_cta == 0 and len(os.listdir(dict_cta)) == len(os.listdir(dict_ct)):
            from format_convert.dcm_np_converter_new import simple_stack_dcm_files
            import numpy as np
            array_non = simple_stack_dcm_files(dict_ct)
            array_cta = simple_stack_dcm_files(dict_cta)
            if np.sum(np.abs(array_cta - array_non)) == 0:
                raise ValueError("CTA and CT is the same")

        if show:
            print("Good pair time!")
            print("The study date is:", study_date_ct)
            print("The acquisition time for CT is:", study_time_ct)
            print("The acquisition time for CTA is:", study_time_cta, 'which is', min_ct_prior_to_cta, 'min after CT')
        return True, "good pair"

    elif min_ct_prior_to_cta < 0:
        print("\n########################")
        print("CTA is collected before CT")
        print("date for ct", study_date_ct, "date for cta", study_date_cta)
        print("time for ct", study_time_ct, "time for cta", study_time_cta)
        print("The acquisition time for CTA is", min_ct_prior_to_cta, 'min after CT')
        print("\n########################")
        return False, "CT-after-CTA"

    elif 60 * 48 > min_ct_prior_to_cta >= 120:
        print("\n########################")
        print("CT and CTA sample_interval time not good")
        print("date for ct", study_date_ct, "date for cta", study_date_cta)
        print("time for ct", study_time_ct, "time for cta", study_time_cta)
        print("The acquisition time for CTA is", min_ct_prior_to_cta, 'min after CT')
        print("\n########################")

        return False, "long_CTA-CT_interval"

    elif min_ct_prior_to_cta >= 60 * 48:
        print("\n########################")
        print("CT and CTA sample_interval time not good")
        print("date for ct", study_date_ct, "date for cta", study_date_cta)
        print("time for ct", study_time_ct, "time for cta", study_time_cta)
        print("The acquisition time for CTA is", min_ct_prior_to_cta, 'min after CT')
        print("\n########################")

        return False, "CTA > 2 days after CT"
    else:
        raise ValueError("Unknown cause of time match")


def check_pair(dict_ct, dict_cta):
    dcm_name_list_ct = os.listdir(dict_ct)
    dcm_name_list_cta = os.listdir(dict_cta)

    dcm_ct = pydicom.read_file(os.path.join(dict_ct, dcm_name_list_ct[0]))
    dcm_cta = pydicom.read_file(os.path.join(dict_cta, dcm_name_list_cta[0]))

    info_key_ct = list(dcm_ct.keys())
    info_key_cta = list(dcm_cta.keys())

    check_key = []
    key_word_list = ["PatientSex", "PatientAge", "PatientID", "PatientName"]
    for key_word in key_word_list:
        if key_word in info_key_ct and key_word in info_key_cta:
            if not dcm_ct[key_word].value == dcm_cta[key_word].value:
                print("information of:", key_word, "is not conform in ct and cta:")
                print("ct info:", dcm_ct[key_word].value, "cta info:", dcm_cta[key_word].value)
                return False
            check_key.append(key_word)
    if len(check_key) < 2:
        print("too little checked keys. check_key:", check_key)
        return False
    return True


def check_resolution(dict_ct, dict_cta):
    resolution_non = get_resolution_from_dcm(dict_ct, show=False)
    resolution_cta = get_resolution_from_dcm(dict_cta, show=False)

    if max(resolution_non) > 2.5:
        print("resolution for non-contrast is low:", resolution_non)
        return False
    if max(resolution_cta) > 2.5:
        print("resolution for CTA is low:", resolution_non)
        return False

    return True


def same_time_close_scanned_name_list():
    def min_prior_dir(dict_ct, dict_cta):
        study_date_ct, study_time_ct = get_study_date_time(dict_ct)
        study_date_cta, study_time_cta = get_study_date_time(dict_cta)

        def get_min_prior(str_a, str_b):
            """
            minutes str_a is prior to str_b
            :param str_a: like '093209'
            :param str_b: like '093539'
            :return: float, minutes str_a is prior to str_b, like 3.5
            """
            prior = (int(str_b[0:2]) - int(str_a[0:2])) * 60 + int(str_b[2:4]) - int(str_a[2:4]) + \
                    (int(str_b[4:6]) - int(str_a[4:6])) / 60

            return prior

        return get_min_prior(study_time_ct, study_time_cta)

    import pe_dataset_management.basic_functions as basic_functions
    name_list = basic_functions.get_all_scan_name()

    print(len(name_list))

    from_nj = []
    other = []

    for name in name_list:
        dir_cta, dir_ct = basic_functions.find_patient_id_dataset_correspondence(name, strip=True)
        dir_dcm = dir_cta.split('/')
        dir_dcm[0] = '/'
        dir_dcm[3] = 'paired_dcm_files'

        dir_cta_t = dir_dcm + [name, 'CTA']
        dir_non_t = dir_dcm + [name, 'non-contrast']

        dir_cta = os.path.join(*dir_cta_t)
        dir_non = os.path.join(*dir_non_t)

        time_prior = min_prior_dir(dir_non, dir_cta)

        if 0 <= time_prior < 5:
            print(name, time_prior)
            if 'N' in name:
                from_nj.append(name)
            else:
                other.append(name)

    print(len(from_nj), len(other))

if __name__ == '__main__':
    shuttle_dcm_pair('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_04_10/NJ1001-1100',
                     dataset_type='Temp', fold=(0, 2))
    exit()
    shuttle_dcm_pair('/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_NJ_24_03_11/NJ0101-300',
                     dataset_type='Temp', fold=(0, 2))
