import os
import numpy as np
from chest_ct_database.feature_manager.add_simulated_non_contrast_from_cta import convert_cta_to_ct
import Tool_Functions.Functions as Functions


save_dict_simulated_non_contrast = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/' \
                                   'simulated_non_contrast_v2/rescaled_ct-denoise'

save_dict_visualization = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/visualization/' \
                          'check_convert_to_non_contrast_v2'

source_dict_cta = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt'

dict_rescaled_cta_denoise = os.path.join(source_dict_cta, 'rescaled_ct-denoise')

dict_semantics = os.path.join(source_dict_cta, 'semantics')


def prepare_material_for_simulate_non_contrast(file_name):
    """

    :param file_name: 'patient-id.npz'
    :return: rescaled_cta-denoise, blood_vessel_mask, heart_mask,
    """

    rescaled_cta_denoise = np.load(os.path.join(dict_rescaled_cta_denoise, file_name))['array']
    heart_mask = np.load(os.path.join(dict_semantics, 'heart_mask', file_name))['array']

    blood_mask = np.load(os.path.join(dict_semantics, 'blood_mask', file_name))['array']
    artery_mask = np.load(os.path.join(dict_semantics, 'artery_mask', file_name))['array']
    vein_mask = np.load(os.path.join(dict_semantics, 'vein_mask', file_name))['array']
    lung_mask = np.load(os.path.join(dict_semantics, 'lung_mask', file_name))['array']

    blood_vessel_mask = np.clip(blood_mask + artery_mask + vein_mask, 0, 1)

    return rescaled_cta_denoise, blood_vessel_mask, heart_mask, lung_mask


def visualize_difference(rescaled_cta_denoise, simulated_non_contrast):

    image_11 = rescaled_cta_denoise[:, :, 256]
    image_12 = simulated_non_contrast[:, :, 256]
    image_1 = np.concatenate([image_11, image_12], axis=0)

    image_21 = rescaled_cta_denoise[:, 256, :]
    image_22 = simulated_non_contrast[:, 256, :]
    image_2 = np.concatenate([image_21, image_22], axis=0)

    image_31 = rescaled_cta_denoise[256, :, :]
    image_32 = simulated_non_contrast[256, :, :]
    image_3 = np.concatenate([image_31, image_32], axis=0)

    image = np.concatenate((image_1, image_2, image_3), axis=1)

    image = np.array(image, 'float32')
    image = np.clip(image, Functions.change_to_rescaled(-1000), Functions.change_to_rescaled(400))

    return image


def process_one_file_name(file_name):
    """

    :param file_name: 'patient-id.npz'
    :return: None
    """
    print("processing:", file_name)

    image_save_path = os.path.join(save_dict_visualization, file_name[:-4] + '.png')

    if os.path.exists(os.path.join(save_dict_simulated_non_contrast, file_name)):
        if os.path.exists(image_save_path):
            print("processed")
            return None

    rescaled_cta_denoise, blood_vessel_mask, heart_mask, lung_mask = \
        prepare_material_for_simulate_non_contrast(file_name)

    simulated_non_contrast_rescaled = convert_cta_to_ct(rescaled_cta_denoise, blood_vessel_mask, heart_mask,
                                                        extra_roi_region=lung_mask,
                                                        broad_cast=True)

    visualize_image = visualize_difference(rescaled_cta_denoise, simulated_non_contrast_rescaled)

    Functions.image_save(visualize_image, image_save_path,
                         gray=True, dpi=300)

    Functions.save_np_array(save_dict_simulated_non_contrast, file_name, simulated_non_contrast_rescaled,
                            compress=True, dtype='float16')


if __name__ == '__main__':
    file_name_list = os.listdir(dict_rescaled_cta_denoise)[::3]
    for fn in Functions.iteration_with_time_bar(file_name_list):
        process_one_file_name(fn)

