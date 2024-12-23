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
import format_convert.dcm_np_converter as convert
import pydicom
import sys
sys.path.append('/home/zhoul0a/Desktop/longxi_platform/')


def check_files_1_to_2(top_dict_patient):
    """
    Step 1 to 2
    1) each patient should has
    a folder: 'CT-not-contrast', a folder: 'CTA-with-contrast-agent', a file 'annotation_for_blood_clot_on_CTA.mha'
    2) the scan time from 'CT-not-contrast' and 'CTA-with-contrast-agent' should be very close
    :param top_dict_patient:
    :param rescaled_ct_save_dict_cta:
    :return: pickle of the resolution of the CTA
    """
    patient_id_list = os.listdir(top_dict_patient)
    patient_id_list.sort()

    num_patients = len(patient_id_list)

    processed_count = 0
    for patient_id in patient_id_list:

        print(patient_id, num_patients - processed_count, 'left')

        top_dict_content = os.path.join(top_dict_patient, patient_id)

        content_name_list = os.listdir(top_dict_content)

        # step 1)
        check_content_name(content_name_list)

        # step 2)
        check_scan_time(os.path.join(top_dict_content, 'CT-not-contrast'),
                        os.path.join(top_dict_content, 'CTA-with-contrast-agent'))

        print("\n\n")

        processed_count += 1


def form_rescaled_cta(top_dict_patient, top_dict_rescaled_cta):
    """
    step 3): for the rescaled ct for CTA
    :param top_dict_patient
    :param top_dict_rescaled_cta:
    :return:
    """
    patient_id_list = os.listdir(top_dict_patient)
    patient_id_list.sort()

    num_patients = len(patient_id_list)

    processed_count = 0
    for patient_id in patient_id_list:

        print(patient_id, num_patients - processed_count, 'left')

        top_dict_content = os.path.join(top_dict_patient, patient_id)

        dict_cta = os.path.join(top_dict_content, 'CTA-with-contrast-agent')

        # step 3)
        rescaled_cta = convert.to_rescaled_ct_for_chest_ct(dict_cta, return_resolution=False)
        Functions.save_np_array(top_dict_rescaled_cta, patient_id + '_CTA.npy', rescaled_cta, compress=False)

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

        rescaled_gt = convert.normalize_gt_array(dict_cta, gt_array_raw)

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
    assert 'CTA-with-contrast-agent' in content_name_list
    assert 'annotation_for_blood_clot_on_CTA.mha' in content_name_list
    assert 'CT-not-contrast' in content_name_list


def check_scan_time(dict_ct, dict_cta):
    dcm_name_list_ct = os.listdir(dict_ct)
    dcm_name_list_cta = os.listdir(dict_cta)

    dcm_ct = pydicom.read_file(os.path.join(dict_ct, dcm_name_list_ct[0]))
    dcm_cta = pydicom.read_file(os.path.join(dict_cta, dcm_name_list_cta[0]))

    if "AcquisitionDateTime" in list(dcm_ct.keys()):

        date_time_ct = dcm_ct["AcquisitionDateTime"].value
        date_time_cta = dcm_cta["AcquisitionDateTime"].value

        study_date_ct = date_time_ct[0:8]  # type is str, like '20200210'
        study_date_cta = date_time_cta[0:8]

    else:
        study_date_ct = dcm_ct["AcquisitionDate"].value  # type is str, like '20200210'
        study_date_cta = dcm_cta["AcquisitionDate"].value

    if not study_date_ct == study_date_cta:
        print("########################")
        print("acquisition date not match")
        print("for ct", study_date_ct, "for cta", study_date_cta)
        print("########################")
    # assert study_date_ct == study_date_cta

    print("The study time is:", study_date_ct)

    study_time_ct = dcm_ct["AcquisitionTime"].value  # type is str, like '093209'

    study_time_cta = dcm_cta["AcquisitionTime"].value

    print("The acquisition time for CT is:", study_time_ct)
    print("The acquisition time for CTA is:", study_time_cta)


if __name__ == '__main__':

    patient_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/dcm_and_gt/'
    gt_save_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_masks/CTA/blood_clot/'
    dict_rescaled_cta = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/CTA/'
    dict_visual_save = '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/blood_clot_gt/'

    rescaled_clot_and_visualize(patient_top_dict, gt_save_top_dict, dict_rescaled_cta, dict_visual_save)

    exit()
    #remove_ds_store_pe()
    # dict_dcm_and_gt = '/Users/richard/Desktop/research projects/pulmonary_embolism/dcm_and_gt/'
    dict_dcm_and_gt = '/home/zhoul0a/Desktop/pulmonary_embolism/dcm_and_gt/'
    # dict_rescaled_cta = '/Users/richard/Desktop/research projects/pulmonary_embolism/rescaled_ct/CTA/'

    form_rescaled_cta(dict_dcm_and_gt, dict_rescaled_cta)
    # check_files_1_to_2(dict_dcm_and_gt)
