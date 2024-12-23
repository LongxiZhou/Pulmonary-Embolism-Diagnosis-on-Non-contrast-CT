import torch
import torch.nn as nn
import pulmonary_embolism_v2.transformer_PE_4D.model_transformer as model_transformer
from pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence import predict_clot_for_sample_sequence, \
    predict_clot_for_rescaled_ct
import Tool_Functions.Functions as Functions
import pulmonary_embolism_v2.sequence_rescaled_ct_converter as converter
import os
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


def predict_on_test_pe_dataset():

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


# usually for visualize CTA
def visualize_from_standard_dataset(dict_rescaled_ct, dict_lung_mask, top_dict_save, interval=2):
    file_name_list = os.listdir(dict_rescaled_ct)
    processed = 0
    for file_name in file_name_list:
        print("processing:", file_name, processed, '/', len(file_name_list))
        rescaled_ct_path = os.path.join(dict_rescaled_ct, file_name)
        rescaled_ct = np.load(rescaled_ct_path)['array']
        rescaled_ct_clip = np.clip(rescaled_ct * 1600 - 600, -200, 450)

        lung_mask = np.load(os.path.join(dict_lung_mask, file_name))['array']
        z_range_array = np.where(lung_mask > 0.1)[2]
        if len(z_range_array) > 0:
            z_min, z_max = np.min(z_range_array), np.max(z_range_array)
        else:
            z_min, z_max = 256, 257

        for i in range(z_min, z_max, interval):
            save_path = os.path.join(top_dict_save, file_name[:-4], str(i))
            # Functions.image_show(predicted_clot_mask[:, :, i])
            Functions.image_save(rescaled_ct_clip[:, :, i], save_path, gray=True, dpi=300)
        processed += 1


# for visualize the predict clot region on non-contrast CT
def visualize_clot_from_sample_sequence_dict(dict_sample_sequence, dict_rescaled_ct, top_dict_save, interval=2):
    file_name_list = os.listdir(dict_sample_sequence)
    processed = 0
    for file_name in file_name_list:
        print("processing:", file_name, processed, '/', len(file_name_list))
        sample_sequence = Functions.pickle_load_object(os.path.join(dict_sample_sequence, file_name))
        rescaled_ct_path = os.path.join(dict_rescaled_ct, file_name[:-7] + '.npz')
        rescaled_ct = np.load(rescaled_ct_path)['array']

        low = -100
        high = 200

        rescaled_ct_clip = np.clip(rescaled_ct * 1600 - 600, low, high)
        predicted_clot_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, (4, 4, 5),
                                                                                     key='clot_prob_mask')
        z_range_array = np.where(predicted_clot_mask > 0.1)[2]
        if len(z_range_array) > 0:
            z_min, z_max = np.min(z_range_array), np.max(z_range_array)
        else:
            z_min, z_max = 256, 257

        rescaled_ct_clip[0, 0, :] = low
        rescaled_ct_clip[-1, -1, :] = high
        predicted_clot_mask[0, 0, :] = 1
        predicted_clot_mask[-1, -1, :] = 0

        for i in range(z_min, z_max, interval):
            save_path = os.path.join(top_dict_save, file_name[:-7], str(i))
            # Functions.image_show(predicted_clot_mask[:, :, i])
            Functions.merge_image_with_mask(rescaled_ct_clip[:, :, i], predicted_clot_mask[:, :, i],
                                            save_path=save_path, show=False, dpi=300)
        processed += 1


# establish sample sequence and predict it
def predict_from_standard_dataset(dict_rescaled_ct, sample_sequence_save_dict, top_dict_database=None, model_path=None):
    file_name_list = os.listdir(dict_rescaled_ct)
    model_loaded = load_saved_model_guided(model_path)

    processed = 0

    for file_name in file_name_list:
        print("processing:", file_name, processed, '/', len(file_name_list))
        rescaled_ct_path = os.path.join(dict_rescaled_ct, file_name)
        sample_sequence_predicted = predict_clot_for_rescaled_ct(rescaled_ct_path, top_dict_database, model_loaded)

        sequence_save_path = os.path.join(sample_sequence_save_dict, file_name[:-4] + '.pickle')

        Functions.pickle_save_object(sequence_save_path, sample_sequence_predicted)
        processed += 1


if __name__ == '__main__':
    visualize_clot_from_sample_sequence_dict(
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/non-contrast/sample_sequence',
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/non-contrast/rescaled_ct',
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/non-contrast/visualization/original')
    exit()

    visualize_from_standard_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/CTA/rescaled_ct-denoise',
                                    '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/CTA/semantics/lung_mask',
                                    '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/CTA/visualization')
    exit()

    predict_from_standard_dataset(
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/non-contrast/rescaled_ct-denoise',
        '/home/zhoul0a/Desktop/pulmonary_embolism/pe_dataset_v2/non-contrast/sample_sequence')
    exit()
    predict_on_test_pe_dataset()
    exit()
    predict_on_single_blind_set(fold=(0, 3))
    exit()
    predict_on_refine_non_pe_test_set()
    exit()
