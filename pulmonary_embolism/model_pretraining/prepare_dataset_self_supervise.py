"""
The dataset is for self supervise or pre-training of the pulmonary embolism model_guided
The dataset is of .pickle files, each is the outputs of functions .slice_cube_sequence.extract_cube_sequence

Pre-requites: the rescaled ct (up-sampled if the thickness is 5mm),
if not have artery mask (with Y shape near the heart), run function "pipeline_segment_artery_and_vein"
"""
import os
import numpy as np
import Tool_Functions.Functions as Functions
import collaborators_package.artery_vein_segmentation.predict as predict_artery_and_vein
import pulmonary_embolism.model_pretraining.slice_cube_sequence as slicer


batch_size = 4  # for segment the artery and vein, need batch_size * 11 GB GPU RAM


def pipeline_segment_artery_and_vein(dict_rescaled_ct, dict_save_artery_mask, dict_save_vein_mask):
    """

    :param dict_rescaled_ct:
    :param dict_save_artery_mask:
    :param dict_save_vein_mask:
    :return:
    """
    array_name_list = os.listdir(dict_rescaled_ct)

    remain_num = len(array_name_list)
    print("there are:", remain_num, 'scans')

    for name in array_name_list:
        print('processing:', name, remain_num, 'scan remains')
        if os.path.exists(os.path.join(dict_save_artery_mask, name[:-4] + '.npz')):
            print("processed")
            remain_num -= 1
            continue

        rescaled_ct = np.load(os.path.join(dict_rescaled_ct, name))

        artery, vein = predict_artery_and_vein.predict_artery_and_vein(rescaled_ct, batch_size)

        Functions.save_np_array(dict_save_vein_mask, name[:-4] + '.npz', vein, compress=True)
        Functions.save_np_array(dict_save_artery_mask, name[:-4] + '.npz', artery, compress=True)

        print("complete")
        remain_num -= 1

    return None


def pipeline_slice_cube_sequence(dict_rescaled_ct, dict_artery_mask, dict_save_min_depth_3, dict_save_min_depth_4,
                                 dict_save_extracted_mask_depth_3=None, dict_save_extracted_mask_depth_4=None):
    """

    :param dict_rescaled_ct:
    :param dict_artery_mask:
    :param dict_save_min_depth_3:
    :param dict_save_min_depth_4:
    :param dict_save_extracted_mask_depth_3:
    :param dict_save_extracted_mask_depth_4:
    :return: None
    """
    array_name_list = os.listdir(dict_rescaled_ct)
    remain_count = len(array_name_list)

    wrong_patient_name_list = ['Scanner-B_B21.npy']
    for name in wrong_patient_name_list:
        if name in array_name_list:
            array_name_list.remove(name)

    for name in array_name_list:
        print('processing:', name, remain_count, 'scan remains')
        if os.path.exists(os.path.join(dict_save_min_depth_4, name[:-4] + '.pickle')):
            print("processed")
            remain_count -= 1
            continue

        rescaled_ct = np.load(os.path.join(dict_rescaled_ct, name))
        artery = np.load(os.path.join(dict_artery_mask, name[:-4] + '.npz'))['array']

        if np.sum(artery) == 0:
            print("this scan is not qualify, scan name:", name)
            remain_count -= 1
            continue

        cube_dict_sequence_3, extracted_mask_3 = slicer.extract_cube_sequence_with_check(
            rescaled_ct, artery, min_depth=3)
        cube_dict_sequence_4, extracted_mask_4 = slicer.extract_cube_sequence_with_check(
            rescaled_ct, artery, min_depth=4)

        if dict_save_min_depth_3 is not None:
            Functions.save_np_array(dict_save_extracted_mask_depth_3, name[:-4], extracted_mask_3, compress=True)
        if dict_save_min_depth_4 is not None:
            Functions.save_np_array(dict_save_extracted_mask_depth_4, name[:-4], extracted_mask_4, compress=True)

        Functions.pickle_save_object(os.path.join(dict_save_min_depth_3, name[:-4] + '.pickle'), cube_dict_sequence_3)
        Functions.pickle_save_object(os.path.join(dict_save_min_depth_4, name[:-4] + '.pickle'), cube_dict_sequence_4)

        remain_count -= 1


if __name__ == '__main__':

    pipeline_slice_cube_sequence('/home/zhoul0a/Desktop/normal_people/rescaled_ct_array/',
                                 '/home/zhoul0a/Desktop/normal_people/rescaled_masks/artery_mask/',
                                 '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_for_normal/min_depth_3/',
                                 '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_for_normal/min_depth_4',
                                 '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/extracted_masks/depth_3/',
                                 '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/extracted_masks/depth_4')
    exit()

    pipeline_slice_cube_sequence(
        '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/normal_scan_extended/',
        '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/artery_mask/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_for_normal/normal_extended/min_depth_3/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_for_normal/normal_extended/min_depth_4/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/extracted_masks/normal_extended/depth_3/',
        '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/extracted_masks/normal_extended/depth_4')

    exit()
    # 92 normal people from 4 different CT scanners.
    pipeline_segment_artery_and_vein('/home/zhoul0a/Desktop/normal_people/rescaled_ct_array/',
                                     '/home/zhoul0a/Desktop/normal_people/rescaled_masks/artery_mask/',
                                     '/home/zhoul0a/Desktop/normal_people/rescaled_masks/vein_mask/')

    exit()
    # 232 patients checked for pulmonary fibrosis, but with negative results.
    pipeline_segment_artery_and_vein(
        '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/normal_scan_extended/',
        '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/artery_mask/',
        '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/vein_mask/')
