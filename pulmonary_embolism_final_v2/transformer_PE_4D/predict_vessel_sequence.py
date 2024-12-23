"""
given a sequence (list of dict)
"""
import med_transformer.utlis as utlis
import torch
import torch.nn as nn
import pulmonary_embolism_v2.transformer_PE_4D.model_transformer as model_transformer
import pulmonary_embolism_v2.sequence_operations.trim_length as trim_sequence_length
import os
import analysis.center_line_and_depth_3D as center_line_and_depth
from pulmonary_embolism_v2.prepare_dataset.get_branch_mask import get_branching_cloud
import basic_tissue_prediction.predict_rescaled as basic_tissue_predict
from pulmonary_embolism_v2.sequence_operations.reduce_bad_scan import func_exclusion_case_dict
from pulmonary_embolism_v2.prepare_dataset.convert_blood_vessel_to_sliced_sequence import convert_ct_into_tubes
import Tool_Functions.Functions as Functions
import numpy as np

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def load_saved_model_guided(model_path=None):
    # load MaskedAutoEncoderModel
    if model_path is None:
        model_path = \
            '/home/zhoul0a/Desktop/pulmonary_embolism/check_points/Simulate_Clot/high_variance_clot/increase_3 (v5)/' \
            'vi_0.014_dice_0.746_precision_phase_model_guided.pth'
    params = {
        # model specifics
        "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
        "in_channel": 1,  # 1 for CT

        "cnn_features": 128,  # number of cnn kernels
        "given_features": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
        "embed_dim": 192,  # the embedding dimension for each input cubes.
        # Require: embed_dim % int(8 * encoder_heads) == 0
        "num_heads": 12,
        "encoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "feature_vector"
        "interaction_depth": 2,
        "decoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "flatten_blood_vessel_mask"
        "segmentation_depth": 1,
        "mlp_ratio": 2.0,

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    model = model_transformer.GuidedWithBranch(
        params["cube_size"], params["in_channel"], params["cnn_features"], params["given_features"],
        params["embed_dim"], params["num_heads"], params["encoding_depth"], params["interaction_depth"],
        params["decoding_depth"], params["segmentation_depth"], params["mlp_ratio"]
    )
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])

    data_dict = torch.load(model_path)
    if type(model) == nn.DataParallel:
        model.module.load_state_dict(data_dict["state_dict"])
    else:
        model.load_state_dict(data_dict["state_dict"])

    return model


def predict_clot_for_sample_sequence(list_sample_sequence, model=None, model_path=None, min_depth=3, trim=True,
                                     trim_length=None):
    """

    :param trim:
    :param min_depth:
    :param list_sample_sequence: list of sample_sequence, or sample_sequence
    :param model: the model_guided on GPU, or None
    :param model_path:
    :return:
    add the key "clot_array" for each sample in list_sample_sequence
    add the key "certainty_array" for each sample in list_sample_sequence
    """
    if trim_length is None:
        trim_length = 3000
    assert len(list_sample_sequence) > 0

    strip = False  # whether input is list of sequence or just sequence

    if type(list_sample_sequence[0]) == dict:
        list_sample_sequence = [list_sample_sequence]
        strip = True

    if trim:
        print("trimming input sequence...")
        list_trimmed_sample_sequence = []
        for sample_sequence in list_sample_sequence:
            trimmed_sample_sequence = trim_sequence_length.reduce_sequence_length(
                sample_sequence, target_length=trim_length, max_branch=9)
            list_trimmed_sample_sequence.append(trimmed_sample_sequence)
        list_sample_sequence = list_trimmed_sample_sequence

    params = {
        # model specifics
        "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
        "in_channel": 1,  # 1 for CT

        "cnn_features": 128,  # number of cnn kernels
        "given_features": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
        "embed_dim": 192,  # the embedding dimension for each input cubes.
        # Require: embed_dim % int(8 * encoder_heads) == 0
        "num_heads": 12,
        "encoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "feature_vector"
        "interaction_depth": 2,
        "decoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "flatten_blood_vessel_mask"
        "segmentation_depth": 1,
        "mlp_ratio": 2.0,

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    if model is None:
        model = load_saved_model_guided(model_path)

    batch_size = len(list_sample_sequence)

    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4, cube_shape, _ = \
            utlis.prepare_tensors_pe_transformer(list_sample_sequence, params["embed_dim"], device='cuda:0',
                                                 training_phase=False, get_flatten_vessel_mask=True,
                                                 guide_depth=min_depth)

        segmentation_before_softmax = model(
            batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4)
        # [B, 2, N, flatten_dim]

        segment_probability_clot = softmax_layer(segmentation_before_softmax)[:, 1, :, :]
        segment_positive_certainty = segmentation_before_softmax[:, 1, :, :] - segmentation_before_softmax[:, 0, :, :]
        # [B, N, flatten_dim]
        # print(torch.min(segment_probability_clot), torch.max(segment_probability_clot))

        segment_probability_clot = \
            utlis.post_process_to_tensor(segment_probability_clot, (5, 5, 5))[:, :, 0, :, :, :].cpu().numpy()
        # [B, N, X, Y, Z]

        segment_positive_certainty = \
            utlis.post_process_to_tensor(segment_positive_certainty, (5, 5, 5))[:, :, 0, :, :, :].cpu().numpy()

        for i in range(batch_size):
            for j in range(len(list_sample_sequence[i])):
                item = list_sample_sequence[i][j]
                item['clot_prob_mask'] = segment_probability_clot[i, j, :, :, :]
                item['clot_certainty_mask'] = segment_positive_certainty[i, j, :, :, :]

    if strip:
        return list_sample_sequence[0]

    return list_sample_sequence


def predict_clot_for_rescaled_ct(rescaled_ct_path, top_dict_database=None, model=None,
                                 model_path=None, min_depth=3, trim=True):
    """

    :param rescaled_ct_path:
    :param top_dict_database:
    :param model: the loaded model for transformer
    :param model_path: transformer model path
    :param min_depth:
    :param trim:
    :return: sample_sequence of the rescaled_ct, added key "clot_array" and "certainty_array"
    """
    rescaled_ct_dict = Functions.get_father_dict(rescaled_ct_path)
    if rescaled_ct_dict[-1] == '/':
        rescaled_ct_dict = rescaled_ct_dict[:-1]
    if top_dict_database is None:
        top_dict_database = Functions.get_father_dict(rescaled_ct_dict)
        dataset_name = ''
    else:
        if top_dict_database[-1] == '/':
            top_dict_database = top_dict_database[:-1]
        split_list_database = top_dict_database.split('/')
        split_list_rescaled_ct_dict = rescaled_ct_dict.split('/')
        if len(split_list_rescaled_ct_dict) == len(split_list_database) + 1:
            dataset_name = ''
        else:
            dataset_name = os.path.join(*split_list_rescaled_ct_dict[(len(split_list_database) + 1)::])

    file_name = rescaled_ct_path.split('/')[-1][:-4]

    semantic_report = Functions.pickle_load_object(
        os.path.join(top_dict_database, 'reports', dataset_name, 'report_dict.pickle'))
    case_dict = semantic_report[file_name]

    func_exclusion_case_dict(case_dict, file_name)  # if not good, it will print out, but still continue analysis

    if rescaled_ct_path[-1] == 'y':
        rescaled_ct = np.load(rescaled_ct_path)
    else:
        rescaled_ct = np.load(rescaled_ct_path)['array']

    blood_vessel_path = os.path.join(top_dict_database, 'semantics', dataset_name, 'blood_mask', file_name + '.npz')

    if os.path.exists(blood_vessel_path):
        vessel_mask = np.load(blood_vessel_path)['array']
    else:
        print("predicting blood vessel mask")
        vessel_mask = basic_tissue_predict.get_prediction_blood_vessel(rescaled_ct, batch_size=2)

    branch_array_path = os.path.join(top_dict_database,
                                     'depth_and_center-line', dataset_name, 'blood_branch_map', file_name + '.npz')

    if os.path.exists(branch_array_path):
        branch_array = np.load(branch_array_path)['array']
    else:
        print("get branching array")
        blood_depth_array = center_line_and_depth.get_surface_distance(vessel_mask)
        blood_center_line_mask = center_line_and_depth.get_center_line(vessel_mask, surface_distance=blood_depth_array)

        branch_array = get_branching_cloud(blood_center_line_mask, blood_depth_array, step=2)
        # step should < cube radius

    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, vessel_mask, absolute_cube_length=(4, 4, 5), only_v1=True,
        branch_array=branch_array)

    return predict_clot_for_sample_sequence(sample_sequence, model, model_path, min_depth, trim)


def predict_on_test_pe_dataset():
    import pulmonary_embolism_v2.sequence_rescaled_ct_converter as converter

    model = load_saved_model_guided()

    top_dict_rescaled_ct = '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset/denoise-rescaled_ct/'
    top_dict_vessel = '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset/semantics/blood_mask/'
    top_dict_vessel_center_line = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset/depth_and_center-line/blood_center_line/'
    top_dict_branch_array = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset/depth_and_center-line/blood_branch_map/'

    top_dict_save = '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/simulate_clot_model_increase_2/v5/'

    fn_list = os.listdir(top_dict_rescaled_ct)

    for file_name in fn_list:
        print(file_name)
        rescaled_ct = np.load(top_dict_rescaled_ct + file_name)['array']
        blood_vessel = np.load(top_dict_vessel + file_name)['array']
        branch_array = np.load(top_dict_branch_array + file_name)['array']
        center_line = np.load(top_dict_vessel_center_line + file_name)['array']

        sequence = \
            converter.extract_sequence_from_rescaled_ct(
                rescaled_ct, blood_vessel_mask=blood_vessel,
                blood_center_line=center_line, branch_array=branch_array, apply_denoise=False)

        sequence_with_clot = predict_clot_for_sample_sequence(sequence, model=model)

        predicted_clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(sequence_with_clot, (4, 4, 5),
                                                                                     key='clot_prob_mask')

        z_range_array = np.where(predicted_clot_mask > 0.3)[2]
        if len(z_range_array) == 0:
            z_min = 257
            z_max = 257
        else:
            z_min, z_max = np.min(z_range_array), np.max(z_range_array)

        print(np.min(predicted_clot_mask), np.max(predicted_clot_mask))

        rescaled_ct_clip = np.clip(rescaled_ct * 1600 - 600, -300, 300)

        rescaled_ct_clip[0, 0, :] = -300
        rescaled_ct_clip[-1, -1, :] = 300
        predicted_clot_mask[0, 0, :] = 1
        predicted_clot_mask[-1, -1, :] = 0

        for i in range(z_min, z_max):
            save_path = top_dict_save + file_name[:-4] + '/' + str(i)
            # Functions.image_show(predicted_clot_mask[:, :, i])
            Functions.merge_image_with_mask(rescaled_ct_clip[:, :, i], predicted_clot_mask[:, :, i],
                                            save_path=save_path, show=False, dpi=300)


def predict_on_refine_non_pe_test_set(all_file=False):
    import pulmonary_embolism_v2.sequence_rescaled_ct_converter as converter

    top_dict_save = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/simulate_clot_model_increase_2/non_clot_orignal/'

    sequence_dataset_dict = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/' \
        'merged-refine_length-3000_branch-7/'

    rescaled_ct_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct/'

    fn_list = os.listdir(rescaled_ct_top_dict)

    sequence_refine_fn_test = []

    sequence_fn_list = os.listdir(sequence_dataset_dict)

    model = load_saved_model_guided()

    for sequence_name in sequence_fn_list:
        if not all_file:
            ord_sum = 0
            for char in sequence_name:
                ord_sum += ord(char)
            if not ord_sum % 5 == 0:
                continue

        if sequence_name[:-7] + '.npz' not in fn_list:
            continue

        sequence_refine_fn_test.append(sequence_name)

    print(len(sequence_refine_fn_test))
    print(sequence_refine_fn_test)

    for sequence_name in sequence_refine_fn_test:
        print(sequence_name)
        sequence = Functions.pickle_load_object(sequence_dataset_dict + sequence_name)

        sequence_with_clot = predict_clot_for_sample_sequence(sequence, model=model)

        predicted_clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(sequence_with_clot, (4, 4, 5),
                                                                                     key='clot_prob_mask')

        rescaled_ct = np.load(rescaled_ct_top_dict + sequence_name[:-7] + '.npz')['array']

        z_range_array = np.where(predicted_clot_mask > 0.3)[2]
        if len(z_range_array) > 0:
            z_min, z_max = np.min(z_range_array), np.max(z_range_array)
        else:
            z_min, z_max = 256, 257

        print(np.min(predicted_clot_mask), np.max(predicted_clot_mask))

        rescaled_ct_clip = np.clip(rescaled_ct * 1600 - 600, -300, 300)

        rescaled_ct_clip[0, 0, :] = -300
        rescaled_ct_clip[-1, -1, :] = 300
        predicted_clot_mask[0, 0, :] = 1
        predicted_clot_mask[-1, -1, :] = 0

        for i in range(z_min, z_max):
            save_path = top_dict_save + sequence_name[:-7] + '/' + str(i)
            # Functions.image_show(predicted_clot_mask[:, :, i])
            Functions.merge_image_with_mask(rescaled_ct_clip[:, :, i], predicted_clot_mask[:, :, i],
                                            save_path=save_path, show=False, dpi=300)


def predict_on_single_blind_set(all_file=True, fold=(0, 1)):
    import pulmonary_embolism_v2.sequence_rescaled_ct_converter as converter

    top_dict_save = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/simulate_clot_model_v3/single_blind/'

    sequence_dataset_dict = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/single_blind_trim/'

    rescaled_ct_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/single_blind_test_dataset/rescaled_ct_denoise/'

    fn_list = os.listdir(rescaled_ct_top_dict)

    sequence_blind_fn_test = []

    sequence_fn_list = os.listdir(sequence_dataset_dict)

    model = load_saved_model_guided()

    for sequence_name in sequence_fn_list[fold[0]::fold[1]]:
        if not all_file:
            ord_sum = 0
            for char in sequence_name:
                ord_sum += ord(char)
            if not ord_sum % 5 == 0:
                continue

        if sequence_name[:-7] + '.npz' not in fn_list:
            continue

        sequence_blind_fn_test.append(sequence_name)

    print(len(sequence_blind_fn_test))
    print(sequence_blind_fn_test)

    processed = 0

    for sequence_name in sequence_blind_fn_test:
        print(sequence_name, processed, '/', len(sequence_blind_fn_test))
        sequence = Functions.pickle_load_object(sequence_dataset_dict + sequence_name)

        sequence_with_clot = predict_clot_for_sample_sequence(sequence, model=model)

        predicted_clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(sequence_with_clot, (4, 4, 5),
                                                                                     key='clot_prob_mask')

        rescaled_ct = np.load(rescaled_ct_top_dict + sequence_name[:-7] + '.npz')['array']

        z_range_array = np.where(predicted_clot_mask > 0.1)[2]
        if len(z_range_array) > 0:
            z_min, z_max = np.min(z_range_array), np.max(z_range_array)
        else:
            z_min, z_max = 256, 257

        print(np.min(predicted_clot_mask), np.max(predicted_clot_mask))

        rescaled_ct_clip = np.clip(rescaled_ct * 1600 - 600, -300, 300)

        rescaled_ct_clip[0, 0, :] = -300
        rescaled_ct_clip[-1, -1, :] = 300
        predicted_clot_mask[0, 0, :] = 1
        predicted_clot_mask[-1, -1, :] = 0

        for i in range(z_min, z_max):
            save_path = top_dict_save + sequence_name[:-7] + '/' + str(i)
            # Functions.image_show(predicted_clot_mask[:, :, i])
            Functions.merge_image_with_mask(rescaled_ct_clip[:, :, i], predicted_clot_mask[:, :, i],
                                            save_path=save_path, show=False, dpi=300)
        processed += 1


if __name__ == '__main__':
    predict_on_test_pe_dataset()
    exit()
    predict_on_single_blind_set(fold=(0, 3))
    exit()

    predict_on_refine_non_pe_test_set()
    exit()
