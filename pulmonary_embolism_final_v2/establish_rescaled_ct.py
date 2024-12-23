"""
each patient has two folder and one mask file
one folder for non-contrast CT
one folder for CT with contrast agent (CTA)
one file for the blood clot mask of the CTA


Check:
1) each patient should has
a folder: 'CT-not-contrast', a folder: 'CTA-with-contrast-agent', a file 'annotation_for_blood_clot_on_CTA.mha'

2) the scan time from 'CT-not-contrast' and 'CTA-with-contrast-agent' should be very close

3) form the rescaled_ct for CTA

4) segment lungs, blood vessels, arteries, veins, to check the quality

5) check the annotation file 'annotation_for_blood_clot_on_CTA.mha', should match blood clot in CTA

6) use the lung mask of the CTA to determine the final resolution for CT

7) form the final rescaled_ct of CT: register the mass center for blood vessel

8) check CT and CTA are from the same patient


"""

import os
import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.dcm_np_converter_new as convert
# from Tool_Functions.file_operations import remove_ds_store_pe
import pydicom
import sys

sys.path.append('/home/zhoul0a/Desktop/longxi_platform/')


def check_files_1_to_2(top_dict_patient, show=True):
    """
    Step 1 to 2
    1) each patient should has
    a folder: 'non-contrast', a folder: 'CTA', a file 'annotation_for_blood_clot_on_CTA.mha' (optional)
    2) the scan time from 'non-contrast' and 'CTA' should be very close for PE patient
    :param show:
    :param top_dict_patient:
    :return: None
    """
    print("########################################")
    print("# Checking Scan Time and Patient Pair...")
    print("########################################\n\n\n")
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
            assert check_scan_time(os.path.join(top_dict_content, 'non-contrast'),
                                   os.path.join(top_dict_content, 'CTA'), show=show)
            assert check_pair(os.path.join(top_dict_content, 'non-contrast'),
                              os.path.join(top_dict_content, 'CTA'))
        if show:
            print("\n")
        processed_count += 1


def form_rescaled_ct_and_cta(top_dict_paired_dcm, top_dict_dataset_cta, top_dict_dataset_non_contrast,
                             high_quality=True):
    """
    step 3): for the rescaled ct for CTA and CT
    :param high_quality:
    :param top_dict_paired_dcm
    :param top_dict_dataset_cta:
    :param top_dict_dataset_non_contrast:
    :return:
    """
    patient_id_list = os.listdir(top_dict_paired_dcm)
    patient_id_list.sort()

    num_patients = len(patient_id_list)

    processed_count = 0
    for patient_id in patient_id_list:

        print(patient_id, num_patients - processed_count, 'left')

        top_dict_content = os.path.join(top_dict_paired_dcm, patient_id)

        dict_cta = os.path.join(top_dict_content, 'CTA')
        dict_ct = os.path.join(top_dict_content, 'non-contrast')

        # step 3)
        print("forming rescaled_ct for non-contrast")
        if not os.path.exists(os.path.join(top_dict_dataset_non_contrast, 'rescaled_ct', patient_id + '.npz')):
            original_resolution_ct = convert.get_resolution_from_dcm(dict_ct)
            if high_quality:
                assert original_resolution_ct[2] <= 2
            rescaled_ct = convert.establish_rescale_chest_ct(
                dict_ct, return_original_resolution=False, original_resolution=original_resolution_ct)
            Functions.save_np_array(os.path.join(top_dict_dataset_non_contrast, 'rescaled_ct'),
                                    patient_id, rescaled_ct, compress=True)

        print("forming rescaled_ct for CTA")
        if not os.path.exists(os.path.join(top_dict_dataset_cta, 'rescaled_ct', patient_id + '.npz')):
            original_resolution_cta = convert.get_resolution_from_dcm(dict_cta)
            assert original_resolution_cta[2] <= 2
            rescaled_cta = convert.establish_rescale_chest_ct(
                dict_cta, return_original_resolution=False, original_resolution=original_resolution_cta)
            Functions.save_np_array(os.path.join(top_dict_dataset_cta, 'rescaled_ct'),
                                    patient_id, rescaled_cta, compress=True)
        print("\n\n")
        processed_count += 1


def segmentation_tissues():
    # step 4) see /home/zhoul0a/Desktop/longxi_platform/basic_tissue_prediction/integrated_segmentation.py
    pass


def rescaled_clot_and_visualize(top_dict_patient, top_dict_save_gt, top_dict_rescaled_cta, top_dict_save_visual):
    # step 5)
    patient_id_list = os.listdir(top_dict_patient)
    patient_id_list.sort()

    num_patients = len(patient_id_list)

    processed_count = 0
    for patient_id in patient_id_list:

        print(patient_id, num_patients - processed_count, 'left')

        top_dict_content = os.path.join(top_dict_patient, patient_id)

        dict_cta = os.path.join(top_dict_content, 'CTA-with-contrast-agent')

        path_mha = os.path.join(top_dict_content, 'annotation_for_blood_clot_on_CTA.mha')

        gt_array_raw = Functions.read_in_mha(path_mha)

        rescaled_gt = convert.establish_rescaled_mask(gt_array_raw, dict_cta)

        Functions.save_np_array(top_dict_save_gt, patient_id + '_CTA.npz', rescaled_gt, compress=True)

        rescaled_cta = np.load(top_dict_rescaled_cta + patient_id + '_CTA.npy')

        rescaled_cta = np.clip(rescaled_cta + 0.5, 0.5, 1.2) / 1.2

        locations_z = list(set(np.where(rescaled_gt > 0.5)[2]))

        print("visualize:", len(locations_z), "images")

        os.makedirs(os.path.join(top_dict_save_visual, patient_id + '/'))

        for z in locations_z:
            image_save_path = os.path.join(top_dict_save_visual, patient_id, patient_id + '_CTA_' + str(z) + '.png')
            merged_image = Functions.merge_image_with_mask(rescaled_cta[:, :, z], rescaled_gt[:, :, z], show=False)
            Functions.image_save(merged_image, image_save_path, dpi=300)

        print("\n\n")

        processed_count += 1


def check_content_name(content_name_list):
    assert 'CTA' in content_name_list
    assert 'non-contrast' in content_name_list
    if len(content_name_list) == 3:
        assert 'annotation_for_blood_clot_on_CTA.mha' in content_name_list
    assert len(content_name_list) < 4


def check_scan_time(dict_ct, dict_cta, show=True):
    dcm_name_list_ct = os.listdir(dict_ct)
    dcm_name_list_cta = os.listdir(dict_cta)

    dcm_ct = pydicom.read_file(os.path.join(dict_ct, dcm_name_list_ct[0]))
    dcm_cta = pydicom.read_file(os.path.join(dict_cta, dcm_name_list_cta[0]))

    # check the date
    if "AcquisitionDateTime" in list(dcm_ct.keys()):

        date_time_ct = dcm_ct["AcquisitionDateTime"].value
        date_time_cta = dcm_cta["AcquisitionDateTime"].value

        study_date_ct = date_time_ct[0:8]  # type is str, like '20200210'
        study_date_cta = date_time_cta[0:8]

    else:
        study_date_ct = dcm_ct["AcquisitionDate"].value  # type is str, like '20200210'
        study_date_cta = dcm_cta["AcquisitionDate"].value

    study_time_ct = dcm_ct["AcquisitionTime"].value  # type is str, like '093209'
    study_time_cta = dcm_cta["AcquisitionTime"].value

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

    if not (study_date_ct == study_date_cta and 0 < get_min_prior(study_time_ct, study_time_cta) < 60):
        if int(study_date_ct) > int(study_date_cta):
            print("########################")
            print("acquisition date time not match")
            print("date for ct", study_date_ct, "date for ct", study_date_ct)
            print("CTA is collected before CT")
            print("########################")
            return False

        if not study_date_ct == study_date_cta:
            print("########################")
            print("acquisition date time not match")
            print("date for ct", study_date_ct, "date for ct", study_date_ct)
            print("########################")

        if not 0 < get_min_prior(study_time_ct, study_time_cta) < 60:
            print("########################")
            print("CT and CTA sample_interval time not good")
            print("The acquisition time for CTA is:", study_time_cta,
                  'which is', min_ct_prior_to_cta, 'min after CT')
            print("########################")

        return False

    if show:
        print("The study date is:", study_date_ct)
        print("The acquisition time for CT is:", study_time_ct)
        print("The acquisition time for CTA is:", study_time_cta, 'which is', min_ct_prior_to_cta, 'min after CT')
    return True


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


def pipeline_pe_high_quality():
    check_files_1_to_2('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_High_Quality', show=False)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/')


def pipeline_normal_high_quality():
    check_files_1_to_2('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_High_Quality', show=False)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality/')


def pipeline_pe_low_quality():
    # CTA and CT sample_interval too large
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'long_CTA-CT_interval/', high_quality=False)

    # CTA and CT sample_interval is good, but the dcm for non-contrast is not good, e.g., too thick, not chest scan,
    # missing slice, etc.
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/', high_quality=False)

    # non-contrast is done before the CTA
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'CT-after-CTA/', high_quality=False)


def pipeline_normal_low_quality():
    # CTA and CT sample_interval too large
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'long_CTA-CT_interval/', high_quality=False)

    # CTA and CT sample_interval is good, but the dcm for non-contrast is not good, e.g., too thick, not chest scan,
    # missing slice, etc.
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/', high_quality=False)

    # non-contrast is done before the CTA
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'CT-after-CTA/', high_quality=False)


def pipeline_pe():
    pipeline_pe_high_quality()
    pipeline_pe_low_quality()


def pipeline_normal():
    pipeline_normal_high_quality()
    pipeline_normal_low_quality()


if __name__ == '__main__':
    # pipeline_pe_high_quality()
    # pipeline_pe_low_quality()
    pipeline_normal()
