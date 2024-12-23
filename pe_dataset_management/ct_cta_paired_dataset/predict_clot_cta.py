import numpy as np
import os
from Tool_Functions import Functions as Functions
from segment_clot_cta.inference.ct_sequence_convert_pe_v3 import extract_sequence_from_rescaled_ct, \
    reconstruct_rescaled_ct_from_sample_sequence
from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided


def get_specific_dict_name(high_resolution, use_denoise, use_simulated_non_contrast, use_blood_high_recall):
    return "(hi-reso, denoise, sim-non, big-roi): " + str(int(high_resolution)) + str(int(use_denoise)) + \
           str(int(use_simulated_non_contrast)) + str(int(use_blood_high_recall))


def load_rescaled_ct(data_top_dict, file_name, use_denoise):
    # load rescaled_ct
    if use_denoise:
        dict_rescaled_ct = os.path.join(data_top_dict, 'rescaled_ct-denoise')
    else:
        dict_rescaled_ct = os.path.join(data_top_dict, 'rescaled_ct')
    assert os.path.exists(dict_rescaled_ct)
    path_ct = os.path.join(dict_rescaled_ct, file_name)
    rescaled_ct = np.load(path_ct)['array']
    return rescaled_ct


def extract_sample_sequence_inference_only(data_top_dict, file_name, high_resolution,
                                           use_denoise, use_simulated_non_contrast, use_blood_high_recall,
                                           return_ct=False):
    # load rescaled_ct
    if use_denoise:
        dict_rescaled_ct = os.path.join(data_top_dict, 'rescaled_ct-denoise')
    else:
        dict_rescaled_ct = os.path.join(data_top_dict, 'rescaled_ct')
    assert os.path.exists(dict_rescaled_ct)
    path_ct = os.path.join(dict_rescaled_ct, file_name)
    rescaled_ct = np.load(path_ct)['array']

    # load depth array and branch map
    if use_simulated_non_contrast:
        dict_center_line_and_depth = os.path.join(data_top_dict, 'simulated_non_contrast', 'depth_and_center-line')
    else:
        dict_center_line_and_depth = os.path.join(data_top_dict, 'depth_and_center-line')
    assert os.path.exists(dict_center_line_and_depth)
    if use_blood_high_recall:
        path_depth_array = os.path.join(dict_center_line_and_depth, 'high_recall_depth_array', file_name)
        path_branch_array = os.path.join(dict_center_line_and_depth, 'high_recall_blood_branch_map', file_name)
    else:
        path_depth_array = os.path.join(dict_center_line_and_depth, 'depth_array', file_name)
        path_branch_array = os.path.join(dict_center_line_and_depth, 'blood_branch_map', file_name)
    depth_array = np.load(path_depth_array)['array']
    branch_array = np.load(path_branch_array)['array']

    # form sample_sequence
    sample_sequence = extract_sequence_from_rescaled_ct(rescaled_ct, depth_array=depth_array, branch_array=branch_array,
                                                        apply_denoise=False, high_resolution=high_resolution,
                                                        strict_trim=False)
    if not return_ct:
        return sample_sequence
    return sample_sequence, rescaled_ct


def predict_and_show(high_resolution=False, file_name='patient-id-135.npz',
                     data_top_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                     image_save_dict='/data_disk/temp/visualize/clot_predict_with_gt/', model_loaded=None,
                     save_dict_clot_mask=None, use_denoise=True, use_simulated_non_contrast=False,
                     use_blood_high_recall=True,
                     save_dict_sample_sequence=None):
    if len(file_name) <= 4:
        file_name = file_name + '.npz'
    if not file_name[-4::] == '.npz':
        file_name = file_name + '.npz'

    dict_save_image_case = os.path.join(image_save_dict, file_name[:-4])
    if os.path.exists(dict_save_image_case):
        print("image path exist at:", dict_save_image_case)
        return None

    if save_dict_sample_sequence is not None:
        specific_dict = get_specific_dict_name(
            high_resolution, use_denoise, use_simulated_non_contrast, use_blood_high_recall)
        sample_save_path = os.path.join(save_dict_sample_sequence, specific_dict, file_name[:-4] + '.pickle')
        if os.path.exists(sample_save_path):
            sample_sequence = Functions.pickle_load_object(sample_save_path)
            rescaled_ct = load_rescaled_ct(data_top_dict, file_name, use_denoise)
        else:
            sample_sequence, rescaled_ct = extract_sample_sequence_inference_only(
                data_top_dict, file_name, high_resolution, use_denoise, use_simulated_non_contrast,
                use_blood_high_recall,
                return_ct=True)
            # save sample sequence
            Functions.pickle_save_object(sample_save_path, sample_sequence)
    else:
        sample_sequence, rescaled_ct = extract_sample_sequence_inference_only(
            data_top_dict, file_name, high_resolution, use_denoise, use_simulated_non_contrast, use_blood_high_recall,
            return_ct=True)

    from segment_clot_cta.inference.inference_pe_v3 import predict_clot_for_sample_sequence
    sample_sequence_with_clot = predict_clot_for_sample_sequence(sample_sequence, trim=False,
                                                                 high_resolution=high_resolution,
                                                                 model=model_loaded)

    if high_resolution:
        absolute_cube_length = (4, 4, 5)
    else:
        absolute_cube_length = (7, 7, 10)

    rescaled_ct = np.clip(rescaled_ct + 0.5, 0.25, 1.2)  # (-1000HU, 520HU)

    predicted_clot_mask = reconstruct_rescaled_ct_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length=absolute_cube_length, key='clot_prob_mask')
    predicted_clot_mask = np.array(predicted_clot_mask > 0.5, 'float32')

    if save_dict_clot_mask is not None:
        Functions.save_np_array(save_dict_clot_mask, file_name, predicted_clot_mask, compress=True)

    roi_region = reconstruct_rescaled_ct_from_sample_sequence(sample_sequence_with_clot, key='depth_cube',
                                                              absolute_cube_length=absolute_cube_length)
    roi_region = np.array(roi_region > 0.5, 'float32')

    z_list = list(set(np.where(predicted_clot_mask > 0.5)[2]))
    z_list.sort()

    for z in z_list[::2]:
        up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_clot_mask[:, :, z], show=False)
        low_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], roi_region[:, :, z], show=False)
        image = np.concatenate((up_image, low_image), axis=0)

        Functions.image_save(image, os.path.join(dict_save_image_case, str(z) + '.png'), dpi=300)


def predict_and_show_dataset(dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                             high_resolution=False, use_denoise=True, use_simulated_non_contrast=True,
                             use_blood_high_recall=True,
                             model_path='/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/'
                                        'gb_0_dice_0.799_precision_phase_model_guided.pth', fold=(0, 1)):

    # the model use this directory to check whether processed
    image_save_dict = os.path.join(dataset_dict, 'visualization', 'predict_clot')

    clot_mask_save_dict = os.path.join(dataset_dict, 'semantics', 'blood_clot')
    save_dict_sample_sequence = os.path.join(dataset_dict, 'sample_sequence', 'pe_v3_inference')

    model = load_saved_model_guided(high_resolution=False, model_path=model_path)

    dict_rescaled_ct = os.path.join(dataset_dict, 'rescaled_ct')
    if not os.path.exists(dict_rescaled_ct):
        print("dataset not exist:", dataset_dict)
        return None

    fn_list = Functions.split_list_by_ord_sum(os.listdir(dict_rescaled_ct), fold=fold)
    processed = 0
    for fn in fn_list:
        print("processing:", fn, processed, '/', len(fn_list))
        predict_and_show(file_name=fn, data_top_dict=dataset_dict, image_save_dict=image_save_dict, model_loaded=model,
                         save_dict_sample_sequence=save_dict_sample_sequence, save_dict_clot_mask=clot_mask_save_dict,

                         high_resolution=high_resolution, use_denoise=use_denoise,
                         use_simulated_non_contrast=use_simulated_non_contrast,
                         use_blood_high_recall=use_blood_high_recall)
        processed += 1


if __name__ == '__main__':
    from pe_dataset_management.basic_functions import get_dataset_relative_path

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    current_fold = (0, 6)
    for dataset in get_dataset_relative_path():
        dataset_dict_cta = os.path.join('/data_disk/CTA-CT_paired-dataset/dataset_CTA', dataset)
        predict_and_show_dataset(dataset_dict=dataset_dict_cta, fold=current_fold)

    exit()
