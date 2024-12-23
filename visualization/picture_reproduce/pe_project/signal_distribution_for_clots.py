import Tool_Functions.Functions as Functions
import numpy as np
import analysis.connect_region_detect as connected_region_detect
from analysis.center_line_and_depth_3D import get_surface_distance
from visualization.visualize_distribution.line_with_confident_interval import create_line
import os


def separate_clot(mask):
    """

    :param mask:
    :return: list of loc_list
    """
    id_loc_list = connected_region_detect.get_sorted_connected_regions(mask, strict=False)
    num_clots = len(id_loc_list)
    if num_clots == 0:
        return []
    largest = len(id_loc_list[1])
    return_list = [id_loc_list[1], ]
    for index in range(2, num_clots + 1):
        if len(id_loc_list[index]) < largest * 0.1:
            break
        return_list.append(id_loc_list[index])
    return return_list


def check_processed(patient_id, top_dict_save):
    fn_list = os.listdir(top_dict_save)
    for fn in fn_list:
        if patient_id[:-4] in fn:
            return True
    return False


def get_clot_signal_distribution(patient_id='patient-id-135.npz', top_dict_save=None):

    if check_processed(patient_id, top_dict_save):
        print(patient_id, 'processed')
        return None
    else:
        print("processing:", patient_id)

    cta = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/'
                  + patient_id)['array']
    cta = Functions.change_to_HU(cta)

    mask = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_gt/'
                   + patient_id)['array']

    list_of_clot_loc_list = separate_clot(mask)

    for clot_loc_list in list_of_clot_loc_list:
        loc_array = Functions.get_location_array(clot_loc_list)
        value_array = cta[loc_array]
        Functions.show_signal_distribution(value_array, number_data_points=30,
                                           save_path=top_dict_save + patient_id[:-4] + '-clot_volume-' +
                                           str(len(clot_loc_list)) + '.svg')
    if len(list_of_clot_loc_list) > 1:
        Functions.show_signal_distribution(cta, mask, number_data_points=30,
                                           save_path=top_dict_save + patient_id[:-4] + '-overall.svg')


def get_clot_signal_depth_distribution(patient_id, top_dict_save=None):

    if check_processed(patient_id, top_dict_save):
        print(patient_id, 'processed')
        return None
    else:
        print("processing:", patient_id)

    cta = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/'
                  + patient_id)['array']
    cta = Functions.change_to_HU(cta)

    mask = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_gt/'
                   + patient_id)['array']

    depth_array_mask = get_surface_distance(mask, strict=True)

    list_of_clot_loc_list = separate_clot(mask)

    for clot_loc_list in list_of_clot_loc_list:
        loc_array = Functions.get_location_array(clot_loc_list)
        value_array_ct = cta[loc_array]
        depth_array_clot = depth_array_mask[loc_array]

        create_line(depth_array_clot, value_array_ct, x_name='clot_depth', y_name='CT value (HU)',
                    show=False,
                    save_path=top_dict_save + patient_id[:-4] + '-clot_volume-' + str(len(clot_loc_list)) + '.svg')

    if len(list_of_clot_loc_list) <= 1:
        return None

    loc_array_overall = np.where(mask > 0)
    value_array_overall = cta[loc_array_overall]
    depth_array_overall = depth_array_mask[loc_array_overall]

    create_line(depth_array_overall, value_array_overall, x_name='clot_depth', y_name='CT value (HU)',
                show=False,
                save_path=top_dict_save + patient_id[:-4] + '-overall.svg')


if __name__ == '__main__':
    patient_id_list = os.listdir(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/')
    top_dict_save_distribution = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                                 'visualization/clot_signal_distribution/'

    top_dict_save_depth_value = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                                'visualization/clot_depth_signal_relationship/'

    current_fold = (0, 5)

    for patient in patient_id_list[current_fold[0]:: current_fold[1]]:
        try:
            get_clot_signal_depth_distribution(patient, top_dict_save_depth_value)
        except:
            print('bad data for:', patient)
        try:
            get_clot_signal_distribution(patient, top_dict_save_distribution)
        except:
            print('bad data for:', patient)