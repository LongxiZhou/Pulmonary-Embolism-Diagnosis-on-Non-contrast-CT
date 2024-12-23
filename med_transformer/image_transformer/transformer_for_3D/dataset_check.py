import visualization.visualize_3d.highlight_semantics as highlight
import visualization.visualize_3d.visualize_stl as stl
import Tool_Functions.Functions as Functions
import os
import numpy as np


def pipeline_process_image_check(num_image_per_scan=15):

    top_dict_mha = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check/vessels/'

    top_dict_rescaled = '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/normal_scan_extended/'
    top_dict_semantic = '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/semantic_1mm/normal_scan_extended/'

    dict_image_save = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/image_check/ct_and_segment_quality/'
    dict_image_good = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/image_check/good_quality/'

    list_file_name = os.listdir(top_dict_rescaled)[::2]

    processed_count = 0
    for file_name in list_file_name:
        print("\nprocessing:", file_name, len(list_file_name) - processed_count, 'left')

        if os.path.exists(dict_image_good + file_name[:-4] + '.png'):
            print('processed')
            processed_count += 1
            continue

        rescaled_ct = np.load(top_dict_rescaled + file_name)
        rescaled_ct = np.clip(rescaled_ct + 0.5, 0, 1)
        rescaled_ct[0, 0, :] = 0
        rescaled_ct[511, 511, :] = 1

        lung_mask = np.load(top_dict_semantic + 'lung_mask/' + file_name[:-4] + '.npz')['array']

        airway_mask = np.load(top_dict_semantic + 'airway_mask/' + file_name[:-4] + '.npz')['array']

        blood_vessel_mask = Functions.read_in_mha(top_dict_mha + file_name[:-4] + '.mha')

        highlighted = highlight.highlight_mask(blood_vessel_mask, rescaled_ct, 'B', transparency=0.25)
        highlighted = highlight.highlight_mask(airway_mask, highlighted, further_highlight=True, transparency=0.25)

        z_start, z_end = Functions.get_bounding_box(lung_mask, pad=2)[2]
        print("z_start, z_end:", z_start, z_end)

        for z in range(z_start, z_end, int((z_end - z_start) / num_image_per_scan)):
            Functions.image_save(highlighted[:, :, z], dict_image_save + file_name[:-4] + '_' + str(z) + '.png',
                                 dpi=300)

        Functions.image_save(highlighted[:, :, 256], dict_image_good + file_name[:-4] + '.png',
                             dpi=300)

        processed_count += 1


def pipeline_check_stl(semantic='blood'):
    vessel_stl_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check/vessels/'
    if semantic == 'airway':
        vessel_stl_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check/airways/'
    file_name_list = os.listdir(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/image_check/good_quality/')
    file_name_list.sort()

    processed = 0
    for file_name in file_name_list:
        print(file_name, "there are", len(file_name_list) - processed, 'left')
        stl.visualize_stl(vessel_stl_dict + file_name[:-4] + '.stl')
        processed += 1


def pipeline_check_deleted_stl():
    vessel_stl_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/stl_check/vessels/'
    file_name_list_good_quality = os.listdir(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/image_check/good_quality/')

    file_name_list_all_file = os.listdir(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/extract_mask_for_check/blood_vessels/'
    )

    file_name_deleted = []

    for file_name in file_name_list_all_file:
        if file_name[:-4] + '.png' not in file_name_list_good_quality:
            file_name_deleted.append(file_name)

    processed = 0
    for file_name in file_name_deleted:
        print(file_name, "there are", len(file_name_deleted) - processed, 'left')
        stl.visualize_stl(vessel_stl_dict + file_name[:-4] + '.stl')
        processed += 1


def pipeline_check_extract_count_mask():
    array = np.load(
        '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/extract_mask_for_check/blood_vessels/Scanner-A_A5.npz')['array']
    Functions.array_stat(array)
    Functions.image_show(array[:, :, 256])


if __name__ == '__main__':
    pipeline_check_extract_count_mask()
    exit()
    pipeline_check_stl('airway')
    exit()
    pipeline_check_deleted_stl()
    exit()

    pipeline_process_image_check()
