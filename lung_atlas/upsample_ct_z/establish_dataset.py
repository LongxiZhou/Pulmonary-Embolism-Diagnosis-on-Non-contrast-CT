"""
establish dataset. Each sample in shape [ct_5_pre + ct_5_mid + ct_5_nex + 5 x ct_1_mid + 5 x penalty, 512, 512]

ct_5 means ct slice with 5 mm thickness; ct_1 means ct slice with 1mm thickness

model_guided input three consecutive ct_5, output up-sampled five ct_1 for the middle ct_5
there are 5 penalty weight array to guide the training, so the shape is [13, 512, 512] for each sample

About the penalty weight array:
We classify each rescaled_ct into six semantic: 1) surface of airway/blood vessel inside lung; 2) airway, blood vessel;
3) airway, blood vessel, lesion; 4) inside the lung or heart; outside the lung and heart
The penalty weight sum for each component is the same; each voxel for same component has the same penalty weight
The final penalty weight is the sum for all_file 5 penalties. Like for voxel on surface of airway, it will have weights
from semantic 1, 2, 3 and 4, while voxel outside lung and heart only benefit from semantic 5.

pre-requite: rescaled_ct, vessel_mask, blood_mask, airway mask, lesion mask
"""
import numpy as np
import Tool_Functions.Functions as Functions
import analysis.get_surface_rim_adjacent_mean as get_surface
import os


def slice_one_rescaled_ct(rescaled_ct, lung_mask, blood_vessel_mask, airway_mask, lesion_mask, heart_mask, interval=5):
    """
    rescaled_ct should be able to get the high quality segmentation for lung mask.
    rescaled_ct should from dcm files with slice thickness <= 1.5 mm
    :param interval: sample_interval for making samples
    :param lesion_mask:
    :param rescaled_ct:
    :param lung_mask:
    :param blood_vessel_mask:
    :param airway_mask:
    :param heart_mask:
    :return: a list of samples, [(mid_z, sample), (mid_z, sample), ...]
    """
    inside_lung = np.where(lung_mask > 0)
    z_min = max(np.min(inside_lung[2]) - 20, 10)
    z_max = min(np.max(inside_lung[2]) + 20, 500)

    blood_airway_combined = np.clip(airway_mask + blood_vessel_mask, 0, 1)

    mask_semantic_1 = get_surface.get_surface(blood_airway_combined, outer=False, strict=False) * lung_mask
    mask_semantic_2 = blood_airway_combined
    mask_semantic_3 = np.array(np.clip(blood_airway_combined + lesion_mask, 0, 1))
    mask_semantic_4 = np.clip(lung_mask + heart_mask, 0, 1)
    mask_semantic_5 = 1 - mask_semantic_4

    semantic_weight_sum = 1000000

    final_penalty_mask = np.array(semantic_weight_sum / np.sum(mask_semantic_1) * mask_semantic_1, 'float32')
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_2) * mask_semantic_2
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_3) * mask_semantic_3
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_4) * mask_semantic_4
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_5) * mask_semantic_5

    def generate_one_sample(mid_z):
        training_sample = np.zeros([13, 512, 512], 'float32')

        # the model_guided inputs, i.e., three ct_5
        training_sample[0, :, :] = np.average(rescaled_ct[:, :, (mid_z - 7): (mid_z - 2)], axis=2)  # ct_5_pre
        training_sample[1, :, :] = np.average(rescaled_ct[:, :, (mid_z - 2): (mid_z + 3)], axis=2)  # ct_5_mid
        training_sample[2, :, :] = np.average(rescaled_ct[:, :, (mid_z + 3): (mid_z + 8)], axis=2)  # ct_5_nex

        # ground truth for the model_guided outputs, i.e., five ct_1
        training_sample[3, :, :] = rescaled_ct[:, :, mid_z - 2]
        training_sample[4, :, :] = rescaled_ct[:, :, mid_z - 1]
        training_sample[5, :, :] = rescaled_ct[:, :, mid_z]
        training_sample[6, :, :] = rescaled_ct[:, :, mid_z + 1]
        training_sample[7, :, :] = rescaled_ct[:, :, mid_z + 2]

        # penalty weights:
        training_sample[8, :, :] = final_penalty_mask[:, :, mid_z - 2]
        training_sample[9, :, :] = final_penalty_mask[:, :, mid_z - 1]
        training_sample[10, :, :] = final_penalty_mask[:, :, mid_z]
        training_sample[11, :, :] = final_penalty_mask[:, :, mid_z + 1]
        training_sample[12, :, :] = final_penalty_mask[:, :, mid_z + 2]

        return training_sample

    training_sample_list = []

    for z in range(z_min, z_max, interval):
        training_sample_list.append((z, generate_one_sample(z)))

    return training_sample_list


def pipeline_process(dict_rescaled_ct, dict_lung_mask, dict_blood_mask, dict_airway_mask, dict_heart_mask,
                     dict_list_lesion, dict_save):
    """

    :param dict_heart_mask:
    :param dict_rescaled_ct:
    :param dict_lung_mask:
    :param dict_blood_mask:
    :param dict_airway_mask:
    :param dict_list_lesion: a list of dict for lesions, like [./infection/, ./nodule_mask/, ...]
    :param dict_save: where to save training samples.
    sample_saved_as: name-rescaled-ct_mid-z.npy, mid-z in four digits, like 0001, 0123
    :return: None
    """
    array_name_list = os.listdir(dict_rescaled_ct)
    print("There are:", len(array_name_list), "rescaled arrays")

    sample_name_list = os.listdir(dict_save)
    processed_array_name_set = set()
    for sample_name in sample_name_list:
        processed_array_name_set.add(sample_name[:-9])
    print(processed_array_name_set)

    processed = 0
    for array_name in array_name_list:
        print("processing:", array_name, processed, '/', len(array_name_list))
        if array_name[:-4] in processed_array_name_set:
            print('processed')
            processed += 1
            continue

        rescaled_ct = np.load(os.path.join(dict_rescaled_ct, array_name))
        lung_mask = np.load(os.path.join(dict_lung_mask, array_name[:-1] + 'z'))['array']
        blood_vessel_mask = np.load(os.path.join(dict_blood_mask, array_name[:-1] + 'z'))['array']
        airway_mask = np.load(os.path.join(dict_airway_mask, array_name[:-1] + 'z'))['array']
        heart_mask = np.load(os.path.join(dict_heart_mask, array_name[:-1] + 'z'))['array']
        lesion_mask = np.zeros([512, 512, 512], 'float32')
        for lesion_dict in dict_list_lesion:
            lesion_mask = lesion_mask + np.load(os.path.join(lesion_dict, array_name[:-1] + 'z'))['array']
        if len(dict_list_lesion) > 0:
            lesion_mask = np.clip(lesion_mask, 0, 1)

        # training_sample_list in [(mid_z, sample), ...]
        training_sample_list = slice_one_rescaled_ct(rescaled_ct, lung_mask, blood_vessel_mask, airway_mask,
                                                     lesion_mask, heart_mask)
        print("there are", len(training_sample_list), 'samples')

        for item in training_sample_list:
            sample_name = array_name[:-4] + '_'
            mid_z = item[0]
            assert 0 <= mid_z < 10000
            if 1000 <= mid_z < 10000:
                sample_name = sample_name + str(mid_z)
            elif 100 <= mid_z < 1000:
                sample_name = sample_name + '0' + str(mid_z)
            elif 10 <= mid_z < 100:
                sample_name = sample_name + '00' + str(mid_z)
            else:
                sample_name = sample_name + '000' + str(mid_z)

            Functions.save_np_array(dict_save, sample_name, item[1], compress=False)

        processed += 1


if __name__ == '__main__':
    pipeline_process('/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/normal_scan_extended/',
                     '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/vessel_mask/',
                     '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/blood_mask/',
                     '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/airway_mask/',
                     '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/heart_mask/',
                     ['/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/infection/',
                      '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/nodule_mask/'],
                     '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/training_samples_extended/')
