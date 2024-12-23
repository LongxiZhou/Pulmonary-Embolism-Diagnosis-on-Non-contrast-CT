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


def predict_file_name(high_resolution=False, file_name='patient-id-135.npz',
                      dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                      model_loaded=None,
                      use_denoise=True, use_simulated_non_contrast=False,
                      use_blood_high_recall=True):
    # form sample_sequence for inference, save to ./sample_sequence/pe_v3_inference_result/conditions/.
    # Then predict clot mask in rescaled array, save to ./semantics/blood_clot/.

    if len(file_name) <= 4:
        file_name = file_name + '.npz'
    if not file_name[-4::] == '.npz':
        file_name = file_name + '.npz'

    path_save_clot = os.path.join(dataset_dict, 'semantics', 'blood_clot', file_name)
    if os.path.exists(path_save_clot):
        print("blood clot mask already exist at:", path_save_clot)
        return None

    save_dict_sample_sequence = os.path.join(dataset_dict, 'sample_sequence', 'pe_v3_inference_result')
    specific_dict = get_specific_dict_name(
        high_resolution, use_denoise, use_simulated_non_contrast, use_blood_high_recall)

    sample_with_clot_save_path = os.path.join(save_dict_sample_sequence, specific_dict, file_name[:-4] + '.pickle')

    if os.path.exists(sample_with_clot_save_path):
        print("sample sequence with clot exists:", path_save_clot)
        return None
    else:
        sample_sequence, rescaled_ct = extract_sample_sequence_inference_only(
            dataset_dict, file_name, high_resolution, use_denoise, use_simulated_non_contrast,
            use_blood_high_recall,
            return_ct=True)

    from segment_clot_cta.inference.inference_pe_v3 import predict_clot_for_sample_sequence
    sample_sequence_with_clot = predict_clot_for_sample_sequence(sample_sequence, trim=False, high_resolution=False,
                                                                 model=model_loaded)

    if high_resolution:
        absolute_cube_length = (4, 4, 5)
    else:
        absolute_cube_length = (7, 7, 10)

    predicted_clot_mask = reconstruct_rescaled_ct_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length=absolute_cube_length, key='clot_prob_mask')
    predicted_clot_mask = np.array(predicted_clot_mask > 0.5, 'float32')

    save_dict_clot_mask = os.path.join(dataset_dict, 'semantics', 'blood_clot')
    if save_dict_clot_mask is not None:
        Functions.save_np_array(save_dict_clot_mask, file_name, predicted_clot_mask, compress=True)

    print("saving sample sequence with clot:", sample_with_clot_save_path)
    Functions.pickle_save_object(sample_with_clot_save_path, sample_sequence_with_clot)


def predict_dataset(dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                    high_resolution=False, use_denoise=True, use_simulated_non_contrast=True,
                    use_blood_high_recall=True, model_path=None, fold=(0, 1)):

    model = load_saved_model_guided(high_resolution=False, model_path=model_path)

    dict_rescaled_ct = os.path.join(dataset_dict, 'rescaled_ct')
    if not os.path.exists(dict_rescaled_ct):
        print("dataset not exist:", dataset_dict)
        return None

    fn_list = Functions.split_list_by_ord_sum(os.listdir(dict_rescaled_ct), fold=fold)
    processed = 0
    for fn in fn_list:
        print("processing:", fn, processed, '/', len(fn_list))
        predict_file_name(high_resolution=high_resolution, file_name=fn, dataset_dict=dataset_dict,
                          model_loaded=model, use_denoise=use_denoise,
                          use_simulated_non_contrast=use_simulated_non_contrast,
                          use_blood_high_recall=use_blood_high_recall)
        processed += 1


def show_file_name(file_name='patient-id-135.npz',
                   dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                   denoise_ct=True, up_only=False, interval=2, top_dict_sample_sequence=None, high_resolution=False):
    if len(file_name) <= 4:
        file_name = file_name + '.npz'
    if not file_name[-4::] == '.npz':
        file_name = file_name + '.npz'

    image_save_dict = os.path.join(dataset_dict, 'visualization', 'predict_clot')
    dict_save_image_case = os.path.join(image_save_dict, file_name[:-4])
    if os.path.exists(dict_save_image_case):
        print("image path exist at:", dict_save_image_case)
        return None

    rescaled_ct = load_rescaled_ct(dataset_dict, file_name, denoise_ct)
    rescaled_ct = np.clip(rescaled_ct + 0.5, 0.25, 1.2)  # (-1000HU, 520HU)

    clot_path = os.path.join(dataset_dict, 'semantics', 'blood_clot', file_name)
    predicted_clot_mask = np.load(clot_path)['array']

    z_list = list(set(np.where(predicted_clot_mask > 0.5)[2]))
    z_list.sort()

    if top_dict_sample_sequence is None:
        top_dict_sample_sequence = os.path.join(dataset_dict, 'sample_sequence', 'pe_v3_inference_result',
                                                '(hi-reso, denoise, sim-non, big-roi): 0111')

    if high_resolution:
        absolute_cube_length = (4, 4, 5)
    else:
        absolute_cube_length = (7, 7, 10)

    if up_only:
        for z in z_list[::interval]:
            up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_clot_mask[:, :, z], show=False)
            Functions.image_save(up_image, os.path.join(dict_save_image_case, str(z) + '.png'), dpi=300)
    else:
        path_sample_sequence = os.path.join(top_dict_sample_sequence, file_name[:-4] + '.pickle')
        sample_sequence = Functions.pickle_load_object(path_sample_sequence)
        roi_region = reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, key='depth_cube',
                                                                  absolute_cube_length=absolute_cube_length)
        roi_region = np.array(roi_region > 0.5, 'float32')
        for z in z_list[::interval]:
            up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_clot_mask[:, :, z], show=False)
            low_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], roi_region[:, :, z], show=False)
            image = np.concatenate((up_image, low_image), axis=0)
            Functions.image_save(image, os.path.join(dict_save_image_case, str(z) + '.png'), dpi=300)


def show_dataset(dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                 denoise_ct=True, up_only=False, interval=2, top_dict_sample_sequence=None, fold=(0, 1),
                 high_resolution=False):
    dict_rescaled_ct = os.path.join(dataset_dict, 'rescaled_ct')
    if not os.path.exists(dict_rescaled_ct):
        print("dataset not exist:", dataset_dict)
        return None

    fn_list = os.listdir(dict_rescaled_ct)[fold[0]:: fold[1]]
    processed = 0
    for fn in fn_list:
        print("processing:", fn, processed, '/', len(fn_list))
        show_file_name(file_name=fn, dataset_dict=dataset_dict, denoise_ct=denoise_ct, up_only=up_only,
                       interval=interval, top_dict_sample_sequence=top_dict_sample_sequence,
                       high_resolution=high_resolution)
        processed += 1


def predict_and_show_dataset(dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                             high_resolution=False, use_denoise=True, use_simulated_non_contrast=True,
                             use_blood_high_recall=True, model_path=None, fold=(0, 1), up_only=False,
                             only_predict=False, only_show=False):

    save_dict_sample_sequence = os.path.join(dataset_dict, 'sample_sequence', 'pe_v3_inference_result')
    specific_dict = get_specific_dict_name(
        high_resolution, use_denoise, use_simulated_non_contrast, use_blood_high_recall)
    save_dict_sample_sequence = os.path.join(save_dict_sample_sequence, specific_dict)

    if only_predict:
        predict_dataset(dataset_dict=dataset_dict, high_resolution=high_resolution, use_denoise=use_denoise,
                        use_simulated_non_contrast=use_simulated_non_contrast,
                        use_blood_high_recall=use_blood_high_recall,
                        model_path=model_path, fold=fold)
        return None

    if only_show:
        show_dataset(dataset_dict=dataset_dict, denoise_ct=use_denoise, up_only=up_only,
                     top_dict_sample_sequence=save_dict_sample_sequence, high_resolution=high_resolution)
        return None

    predict_dataset(dataset_dict=dataset_dict, high_resolution=high_resolution, use_denoise=use_denoise,
                    use_simulated_non_contrast=use_simulated_non_contrast,
                    use_blood_high_recall=use_blood_high_recall,
                    model_path=model_path, fold=fold)
    show_dataset(dataset_dict=dataset_dict, denoise_ct=use_denoise, up_only=up_only,
                 top_dict_sample_sequence=save_dict_sample_sequence, fold=fold, high_resolution=high_resolution)


if __name__ == '__main__':
    from pe_dataset_management.basic_functions import get_dataset_relative_path

    model_best = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/' \
                 'gb_0_dice_0.799_precision_phase_model_guided.pth'

    current_fold = (0, 2)
    Functions.set_visible_device('1')

    predict_and_show_dataset(
        dataset_dict='/data_disk/RSNA-PE_dataset', fold=current_fold,
        model_path=model_best, use_denoise=False, use_simulated_non_contrast=False, use_blood_high_recall=False,
        high_resolution=False, only_predict=True
    )
    exit()

    for dataset in get_dataset_relative_path():
        print(dataset)
        predict_and_show_dataset(
            dataset_dict=os.path.join('/data_disk/CTA-CT_paired-dataset/dataset_CTA', dataset), fold=current_fold,
            model_path=model_best, only_predict=True)  # show will spend on time even for processed check

    exit()

    predict_and_show_dataset(
        dataset_dict='/data_disk/Altolia_share/PENet_dataset', fold=current_fold,
        model_path=model_best, use_denoise=True, use_simulated_non_contrast=True, use_blood_high_recall=True
    )
    exit()
