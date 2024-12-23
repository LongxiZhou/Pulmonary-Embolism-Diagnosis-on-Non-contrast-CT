import torch
import med_transformer.image_transformer.transformer_for_3D.training_iterations_v4 as train_iterations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {
    "resume": True,

    "n_epochs": 450,

    "lr": 1e-3,
    "batch_ct": 4,  # number of ct in each batch,    2 for lung region, 2 for all_file ct,
    "batch_ct_test": 4,
    "batch_split": 1,  # each ct scan split into how many samples
    "ratio_mask_input": 0,  # the ratio masked   0.85 for lung regions, 0.9 for lung_heart,  0.975 for all_file ct
    "ratio_predict": 1,  # the ratio to predict  0.15 for lung regions, 0.1 for lung_heart,  0.025 for all_file ct

    # prepare dataset (pick samples from lists and form tensors) can be time consuming
    "num_prepared_dataset_train": 1,  # number of prepared dataset during training
    "num_prepared_dataset_test": 1,  # number of prepared dataset during testing, higher the more accuracy, but slower
    "reuse_count": 18,  # number of times each dataset be used

    "tissue_weight_tuple": (1, 1, 1, 1),  # penalty weight for the loss function,
    # see "get_penalty_array" in .transformer_for_3D.convert_ct_to_sliced_sequence.py for channel information
    # channel 0 for blood vessels, channel 1 for airways, channel 2 for pulmonary parenchyma, channel 3 for others

    "input_output_overlap": True,  # whether the query can be a part of the input
    "penalty_padding_cube": None,  # the default penalty weight array for padding cubes

    "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
    "in_channel": 1,  # 1 for CT
    "embed_dim": 384,  # the embedding dimension for each input cubes.
    # Require: embed_dim % int(6 * encoder_heads) == 0
    "given_dim": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
    "encoder_depth": 2,  # how many encoder encoding_blocks for encoding
    "encoder_heads": 16,  # for each encoder, how many attention heads
    "decoder_embed_dim": 384,  # the embedding dimension for decoding phase.
    # Require: decoder_embed_dim % decoder_num_heads == 0
    "decoder_depth": 1,  # how many decoder encoding_blocks
    "extra_decoder_depth": 1,
    # check the model_guided version in "transformer_for_3D.training_iterations_focal"
    "decoder_heads": 16,  # for each decoder, how many attention heads
    "mlp_ratio": 2.0,  # the DNN setting: len(vector) -> int(mlp_ratio * len(vector)) -> len(vector)

    'num_workers': 24,  # num CPU for the parallel data loading
    'min_parallel_ct': None,  # the upper limit for CPU parallel is min_parallel_ct * batch_split, None means Inf
    "drop_last": False,  # for each epoch, whether to neglect the last few samples that < batch_ct during Training
    "pin_memory": True,  # store the data in the CPU ram

    #"train_data_dir": "/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_healthy/blood_vessels/",
    #"test_data_dir": "/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/list_pickle_dataset_healthy/blood_vessels/",

    "train_data_dir": "/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/blood_vessel_merge/",
    "test_data_dir": "/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/blood_vessel_merge/",

    "checkpoint_dir": "/home/zhoul0a/Desktop/pulmonary_embolism/check_point_guide/training/identity_transform/",

    "saved_model_filename": "model_guided.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_file_name": None,  # the list of file name to remove
}


train_iterations.training(parameters)

exit()

for difficulty in range(10):

    parameters["ratio_mask_input"] = difficulty / 18
    print(parameters)
    train_iterations.training(parameters)

exit()
ct_name_good_quality = os.listdir(
    '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_normal_v2/dataset_check/image_check/good_quality/')
ct_name_all = os.listdir(parameters['train_data_dir'])

all_ct = False
if all_ct:  # cannot feed to0 much data so separate the dataset
    for swap_count in range(1, 20):

        print("#####################")
        print("swap:", swap_count)

        wrong_file_name_list = []

        for name in ct_name_all:
            png_name = name[:-7] + '.png'
            if png_name not in ct_name_good_quality:
                wrong_file_name_list.append(name)

        # for half of the dataset
        for name in ct_name_all[int(swap_count % 2)::2]:
            wrong_file_name_list.append(name)
        wrong_file_name_list = list(set(wrong_file_name_list))

        parameters["wrong_file_name"] = wrong_file_name_list
        parameters["tissue_weight_tuple"] = (20, 20, 1 + swap_count / 4, 10)

        train_iterations.training(parameters)

else:
    wrong_file_name_list = []
    for name in ct_name_all:
        png_name = name[:-7] + '.png'
        if png_name not in ct_name_good_quality:
            wrong_file_name_list.append(name)
    parameters["wrong_file_name"] = wrong_file_name_list
    train_iterations.training(parameters)
