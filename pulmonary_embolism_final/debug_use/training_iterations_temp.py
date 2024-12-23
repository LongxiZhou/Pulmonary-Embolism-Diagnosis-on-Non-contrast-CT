"""
Call function "training" to start the training.
"""

import os
import torch
import pulmonary_embolism_final.training.dataset_and_loader as dataset_and_loader
import pulmonary_embolism_final.utlis.sample_to_tensor_with_annotation as sample_to_tensor_with_annotation
import pulmonary_embolism_final.utlis.sample_to_tensor_simulate_clot as sample_to_tensor_simulate_clot


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    clot_dataset = dataset_and_loader.ClotDataset(params["top_dict_clot_pickle"], mode=params['mode'])

    non_pe_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list_non_pe"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    pe_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list_pe"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    test_loader_non_pe = dataset_and_loader.DataLoaderSimulatedClot(
        clot_dataset, non_pe_sample_dataset_test, params["batch_size_simu"], shuffle=False,
        num_workers=params["num_workers_simu"], mode="test", clot_volume_range=params["clot_volume_range"],
        min_clot=params["min_clot"], num_clot_each_sample_range=params["num_clot_each_sample_range"],
        augment=params["augment"], num_prepared_dataset_test=params["num_prepared_dataset_test"],
        embed_dim=params["embed_dim"], sample_sequence_length=params["sample_sequence_length"])

    test_loader_pe = dataset_and_loader.DataLoaderWithAnnotation(
        pe_sample_dataset_test, params["batch_size_with_gt"], num_workers=params["num_workers_with_gt"],
        augment=params["augment"], random_select=False, shuffle=True, embed_dim=params["embed_dim"],
        sample_sequence_length=params["sample_sequence_length"])

    check_penalty(test_loader_non_pe, test_loader_pe)


def check_penalty(test_loader_non_pe, test_loader_pe):
    with torch.no_grad():
        for i, batch_sample in enumerate(test_loader_non_pe):

            array_packages_a, list_sample_attention_a = batch_sample
            tensors_a = sample_to_tensor_simulate_clot.put_arrays_on_device_simu_clot(
                array_packages_a, device='cuda:0', training_phase=True, penalty_normalize_func=None)

            array_packages_b, list_sample_attention_b = test_loader_pe.extract_data_from_sub_process_and_start_a_new()
            tensors_b = sample_to_tensor_with_annotation.put_arrays_on_device_with_gt(
                array_packages_b, device='cuda:0', training_phase=True, penalty_normalize_func=None)

            (batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region,
             cube_shape, clot_gt_tensor, penalty_weight_tensor), list_sample_attention = \
                dataset_and_loader.merge_tensor_packages(
                    (tensors_a, list_sample_attention_a), (tensors_b, list_sample_attention_b))

            print(i)
            print("\n\nsimulate clot")
            show_details_tensor(tensors_a)

            print("\n\nwith annotation")
            show_details_tensor(tensors_b)

            print('\n\n')
            print(clot_gt_tensor.shape)
            print(list_sample_attention)

            if i + 1 == len(test_loader_non_pe):
                test_loader_pe.clear_sub_process_queue()


def show_details_tensor(tensors):
    batch_tensor, pos_embed_tensor, given_vector, flatten_roi, cube_shape, \
        clot_gt_tensor, penalty_weight_tensor = tensors
    batch_size = batch_tensor.shape[0]
    print("shape batch_tensor:", batch_tensor.shape)
    print("shape clot_gt_tensor", clot_gt_tensor.shape)   # [B, 2, N, flatten_dim]
    # clot_gt_tensor = torch.stack((clot_gt_tensor_negative, clot_gt_tensor_positive), dim=1)

    print("shape penalty_weight_tensor", penalty_weight_tensor.shape)  # [B, 2, N, flatten_dim]
    # penalty_weight_tensor = torch.stack((penalty_weight_fp, penalty_weight_fn), dim=1)

    print("gt sum_negative:", torch.sum(clot_gt_tensor[:, 0]) / batch_size)
    print("gt sum_positive:", torch.sum(clot_gt_tensor[:, 1]) / batch_size)
    print("penalty_sum fp:", torch.sum(penalty_weight_tensor[:, 0]) / batch_size)
    print("penalty_sum fn:", torch.sum(penalty_weight_tensor[:, 1]) / batch_size)
