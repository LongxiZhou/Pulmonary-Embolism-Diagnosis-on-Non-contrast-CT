"""
Add simulated non-contrast for CTA scans
"""
import analysis.get_surface_rim_adjacent_mean as get_surface
import chest_ct_database.feature_manager.basic_funcs_add_features as add_features
import Tool_Functions.Functions as Functions
import os
import numpy as np


def convert_cta_to_ct(rescaled_ct, blood_vessel_mask=None, heart_mask=None, extra_roi_region=None, broad_cast=True):
    """

    :param broad_cast:
    :param heart_mask:
    :param blood_vessel_mask:
    :param rescaled_ct:
    :param extra_roi_region: the region to reset signal, e.g., lung mask
    :return:
    """
    # cover all blood region influenced by CTA
    if blood_vessel_mask is None or heart_mask is None:
        import basic_tissue_prediction.predict_rescaled as predictor
        if blood_vessel_mask is None:
            blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct)
        if heart_mask is None:
            heart_mask = predictor.predict_heart_rescaled_array(rescaled_ct)

    blood_vessel_and_heart_region = np.array(blood_vessel_mask + heart_mask > 0, 'float32')

    high_recall_blood_heart_mask = blood_vessel_and_heart_region + get_surface.get_surface(
        blood_vessel_mask, outer=True, strict=False)
    # you can add more if the mask recall is not high
    high_recall_blood_heart_mask = high_recall_blood_heart_mask + get_surface.get_surface(
        high_recall_blood_heart_mask, outer=True, strict=False)

    if extra_roi_region is not None:
        high_recall_blood_heart_mask = high_recall_blood_heart_mask + extra_roi_region

    if broad_cast:
        from analysis.connect_region_detect import propagate_to_wider_region
        valid_region = np.array(rescaled_ct > Functions.change_to_rescaled(100), 'float32')
        valid_region = valid_region - get_surface.get_surface(valid_region, outer=False, strict=False)
        valid_region = valid_region - get_surface.get_surface(valid_region, outer=False, strict=False)
        valid_region = valid_region - get_surface.get_surface(valid_region, outer=False, strict=False)
        seed_region = valid_region * high_recall_blood_heart_mask
        broad_cast_region = propagate_to_wider_region(valid_region, seed_region, strict=True, return_id_loc_dict=False)
        high_recall_blood_heart_mask = high_recall_blood_heart_mask + broad_cast_region

    if extra_roi_region is not None or broad_cast:
        high_recall_blood_heart_mask = np.clip(high_recall_blood_heart_mask, 0, 1)

    # you can add more if the mask recall is not high
    high_recall_blood_heart_mask = high_recall_blood_heart_mask + get_surface.get_surface(
        high_recall_blood_heart_mask, outer=True, strict=False)
    high_recall_blood_heart_mask = high_recall_blood_heart_mask + get_surface.get_surface(
        high_recall_blood_heart_mask, outer=True, strict=False)
    high_recall_blood_heart_mask = high_recall_blood_heart_mask + get_surface.get_surface(
        high_recall_blood_heart_mask, outer=True, strict=False)

    # this region all not change CT value
    intact_region = np.array(rescaled_ct < Functions.change_to_rescaled(100), 'int8') + 1 - high_recall_blood_heart_mask
    intact_region = np.array(intact_region > 0, 'float32')

    intact_signals = rescaled_ct * intact_region

    # this region all change CT value. set to 100 HU:
    modify_region = 1 - intact_region

    modified_signals = 100

    modified_signals = Functions.change_to_rescaled(modified_signals)

    modified_signals = modified_signals * modify_region

    new_rescaled_ct = intact_signals + modified_signals

    return new_rescaled_ct


def func_file_operation(list_top_dict_reference, dataset_sub_dir, file_name):
    """

    get denoise rescaled ct

    :param list_top_dict_reference: [top_dict_rescaled_ct-denoise, top_dict_semantics]
    :param dataset_sub_dir:
    :param file_name:
    :return: simulated_non_contrast_rescaled, visualize_image
    """

    file_path_denoise_ct = os.path.join(list_top_dict_reference[0], dataset_sub_dir, file_name)
    dict_semantics = os.path.join(list_top_dict_reference[1], dataset_sub_dir)

    print("loading:", file_path_denoise_ct)

    if file_path_denoise_ct[-1] == 'z':
        assert file_path_denoise_ct[:-4] + '.npz' == file_path_denoise_ct
        rescaled_cta_denoise = np.load(file_path_denoise_ct)['array']
    else:
        assert file_path_denoise_ct[:-4] + '.npy' == file_path_denoise_ct
        rescaled_cta_denoise = np.load(file_path_denoise_ct)

    heart_mask = np.load(os.path.join(dict_semantics, 'heart_mask', file_name))['array']
    lung_mask = np.load(os.path.join(dict_semantics, 'lung_mask', file_name))['array']

    path_high_recall_blood_mask = os.path.join(dict_semantics, 'blood_mask_high_recall', file_name)
    if os.path.exists(path_high_recall_blood_mask):
        blood_vessel_mask = np.load(path_high_recall_blood_mask)['array']
    else:
        print("high recall mask not processed, preparing high recall mask")
        blood_mask = np.load(os.path.join(dict_semantics, 'blood_mask', file_name))['array']
        artery_mask = np.load(os.path.join(dict_semantics, 'artery_mask', file_name))['array']
        vein_mask = np.load(os.path.join(dict_semantics, 'vein_mask', file_name))['array']
        blood_vessel_mask = np.clip(blood_mask + artery_mask + vein_mask, 0, 1)
        blood_vessel_mask = blood_vessel_mask * np.array(
            rescaled_cta_denoise > Functions.change_to_rescaled(-200), 'float32')
        Functions.save_np_array(
            os.path.join(dict_semantics, 'blood_mask_high_recall'), file_name, blood_vessel_mask, compress=True)

    simulated_non_contrast_rescaled = convert_cta_to_ct(rescaled_cta_denoise, blood_vessel_mask, heart_mask,
                                                        extra_roi_region=lung_mask,
                                                        broad_cast=True)

    visualize_image = visualize_difference(rescaled_cta_denoise, simulated_non_contrast_rescaled)
    return simulated_non_contrast_rescaled, visualize_image


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


def func_file_save(save_dict, file_name, feature_package):
    simulated_non_contrast_rescaled, visualize_image = feature_package

    save_dict_rescaled_ct = os.path.join(save_dict, 'rescaled_ct-denoise')
    save_dict_visualization = os.path.join(save_dict, 'visualization/check_convert_to_non_contrast')

    save_path_image = os.path.join(save_dict_visualization, file_name[:-4] + '.png')
    Functions.image_save(visualize_image, save_path_image, dpi=300, gray=True)
    Functions.save_np_array(save_dict_rescaled_ct, file_name[:-4] + '.npz',
                            simulated_non_contrast_rescaled, compress=True, dtype='float16')


def func_check_processed(save_dict, file_name):
    save_dict_rescaled_ct = os.path.join(save_dict, 'rescaled_ct-denoise')
    path_saved = os.path.join(save_dict_rescaled_ct, file_name[:-4] + '.npz')

    save_dict_visualization = os.path.join(save_dict, 'visualization/check_convert_to_non_contrast')
    save_path_image = os.path.join(save_dict_visualization, file_name[:-4] + '.png')

    if os.path.exists(path_saved) and os.path.exists(save_path_image):
        return True
    return False


def add_simulated_non_contrast(top_dict_rescaled_ct_denoise, top_dict_semantics, top_dict_save, fold=(0, 1)):
    func_file_operation_semantic = func_file_operation
    add_features.func_add_feature(top_dict_rescaled_ct_denoise, [top_dict_rescaled_ct_denoise, top_dict_semantics],
                                  top_dict_save,
                                  func_file_operation_semantic,
                                  func_file_save, func_check_processed=func_check_processed, fold=fold)


if __name__ == '__main__':
    current_fold = (0, 10)
    Functions.set_visible_device('1')
    top_dict = '/data_disk/CTA-CT_paired-dataset/dataset_CTA/'
    add_simulated_non_contrast(top_dict + 'rescaled_ct-denoise/',
                               top_dict + 'semantics/',
                               top_dict + 'simulated_non_contrast/',
                               fold=current_fold)
