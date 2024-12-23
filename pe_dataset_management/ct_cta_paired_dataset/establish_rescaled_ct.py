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
import sys
import Tool_Functions.file_operations as file_operations
# from pe_dataset_management.ct_cta_paired_dataset.classify_quality_for_dcm_pairs import check_files_1_to_2

sys.path.append('/home/zhoul0a/Desktop/longxi_platform/')


def form_rescaled_ct_and_cta(top_dict_paired_dcm, top_dict_dataset_cta, top_dict_dataset_non_contrast,
                             high_quality=True, fold=(0, 1)):
    """
    step 3): for the rescaled ct for CTA and CT
    :param fold:
    :param high_quality:
    :param top_dict_paired_dcm
    :param top_dict_dataset_cta:
    :param top_dict_dataset_non_contrast:
    :return:
    """

    patient_id_list = os.listdir(top_dict_paired_dcm)
    patient_id_list = Functions.split_list_by_ord_sum(patient_id_list, fold=fold)

    num_patients = len(patient_id_list)

    wrong_list = ['Z249', 'patient-id-p105', 'patient-id-p76', 'patient-id-p77', 'patient-id-p80', 'patient-id-p82',
                  'patient-id-p83', 'patient-id-p85', 'patient-id-p87', 'patient-id-p89', 'patient-id-p92',
                  'patient-id-p93', 'patient-id-p97', 'liangrenhe', 'yangshixiu']

    processed_count = 0
    for patient_id in patient_id_list:

        print(patient_id, num_patients - processed_count, 'left')
        if patient_id in wrong_list:
            print("wrong scan cannot establish rescaled ct")
            processed_count += 1
            continue

        top_dict_content = os.path.join(top_dict_paired_dcm, patient_id)

        dict_cta = os.path.join(top_dict_content, 'CTA')
        dict_ct = os.path.join(top_dict_content, 'non-contrast')

        # step 3)

        try:
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
                if high_quality:
                    assert original_resolution_cta[2] <= 2
                rescaled_cta = convert.establish_rescale_chest_ct(
                    dict_cta, return_original_resolution=False, original_resolution=original_resolution_cta)
                Functions.save_np_array(os.path.join(top_dict_dataset_cta, 'rescaled_ct'),
                                        patient_id, rescaled_cta, compress=True)
        except:
            target_dict = Functions.get_father_dict(top_dict_paired_dcm)
            current_category = top_dict_paired_dcm.split('/')[-1]
            i = -2
            while len(current_category) == 0:
                current_category = top_dict_paired_dcm.split('/')[i]
                i = i - 1
            target_dict = os.path.join(target_dict, 'waiting_to_recheck', current_category,  patient_id)
            file_operations.move_file_or_dir(top_dict_content, target_dict)

            file_operations.remove_path_or_directory(
                os.path.join(top_dict_dataset_non_contrast, 'rescaled_ct', patient_id + '.npz'))
            file_operations.remove_path_or_directory(
                os.path.join(top_dict_dataset_cta, 'rescaled_ct', patient_id + '.npz'))

        print("\n\n")
        processed_count += 1


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


def pipeline_pe_high_quality(fold=(0, 1)):
    # check_files_1_to_2('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_High_Quality', show=False, semantic='PE')
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/', fold=fold)


def pipeline_normal_high_quality(fold=(0, 1)):
    # check_files_1_to_2('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_High_Quality',
    #                    show=False, semantic='Normal')
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_High_Quality/', fold=fold)


def pipeline_pe_low_quality(fold=(0, 1)):
    # CTA and CT sample_interval too large
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'long_CTA-CT_interval/', high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'CTA > 2 days after CT/', high_quality=False, fold=fold)

    # CTA and CT sample_interval is good, but the dcm for non-contrast is not good, e.g., too thick, not chest scan,
    # missing slice, etc.
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/', high_quality=False, fold=fold)

    # non-contrast is done before the CTA
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/PE_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_Low_Quality/'
                             'CT-after-CTA/', high_quality=False, fold=fold)


def pipeline_normal_low_quality(fold=(0, 1)):
    # CTA and CT sample_interval too large
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'long_CTA-CT_interval/', high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'CTA > 2 days after CT/', high_quality=False, fold=fold)

    # CTA and CT sample_interval is good, but the dcm for non-contrast is not good, e.g., too thick, not chest scan,
    # missing slice, etc.
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/', high_quality=False, fold=fold)

    # non-contrast is done before the CTA
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Normal_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Normal_Low_Quality/'
                             'CT-after-CTA/', high_quality=False, fold=fold)


def pipeline_pe(fold=(0, 1)):
    pipeline_pe_high_quality(fold=fold)
    pipeline_pe_low_quality(fold=fold)


def pipeline_normal(fold=(0, 1)):
    pipeline_normal_high_quality(fold=fold)
    pipeline_normal_low_quality(fold=fold)


def pipeline_temp_high_quality(fold=(0, 1)):
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_High_Quality/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Temp_High_Quality/', fold=fold)


def pipeline_temp_low_quality(fold=(0, 1)):
    # CTA and CT sample_interval too large
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_Low_Quality/'
                             'long_CTA-CT_interval/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Temp_Low_Quality/'
                             'long_CTA-CT_interval/', high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_Low_Quality/'
                             'CTA > 2 days after CT/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Temp_Low_Quality/'
                             'CTA > 2 days after CT/', high_quality=False, fold=fold)

    # CTA and CT sample_interval is good, but the dcm for non-contrast is not good, e.g., too thick, not chest scan,
    # missing slice, etc.
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Temp_Low_Quality/'
                             'good_CTA-CT_interval_but_bad_dcm/', high_quality=False, fold=fold)

    # non-contrast is done before the CTA
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/Temp_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_Low_Quality/'
                             'CT-after-CTA/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/Temp_Low_Quality/'
                             'CT-after-CTA/', high_quality=False, fold=fold)


def pipeline_temp(fold=(0, 1)):
    pipeline_temp_high_quality(fold=fold)
    pipeline_temp_low_quality(fold=fold)


def pipeline_may_not_pair(fold=(0, 1)):
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/may_not_pair/Normal/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/may_not_pair/Normal/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/may_not_pair/Normal/',
                             high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/may_not_pair/PE/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/may_not_pair/PE/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/may_not_pair/PE/',
                             high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/may_not_pair/Temp/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/may_not_pair/Temp/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/may_not_pair/Temp/',
                             high_quality=False, fold=fold)
    
    
def pipeline_strange_data(fold=(0, 1)):
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/strange_data/Normal/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/strange_data/Normal/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/strange_data/Normal/',
                             high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/strange_data/PE/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/strange_data/PE/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/strange_data/PE/',
                             high_quality=False, fold=fold)
    form_rescaled_ct_and_cta('/data_disk/CTA-CT_paired-dataset/paired_dcm_files/strange_data/Temp/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_CTA/strange_data/Temp/',
                             '/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/strange_data/Temp/',
                             high_quality=False, fold=fold)


def pipeline_all(fold=(0, 1)):
    pipeline_temp(fold=fold)
    pipeline_pe_high_quality(fold=fold)
    pipeline_pe_low_quality(fold=fold)
    pipeline_normal(fold=fold)


if __name__ == '__main__':
    #
    # pipeline_temp()
    # pipeline_pe_high_quality()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
    pipeline_all(fold=(0, 4))
    # pipeline_pe_low_quality(fold=(0, 6))
    # pipeline_normal()
