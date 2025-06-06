"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import med_transformer.image_transformer.transformer_for_PE.model_mae as model_mae
import med_transformer.image_transformer.transformer_for_PE.loss_functions as loss_function
import med_transformer.image_transformer.transformer_for_PE.dataset_and_loader as dataset_and_loader
import med_transformer.utlis as utlis
from med_transformer.building_blocks import flatten_batch


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True):
    if not best:  # this means we store the current model_guided
        filename = "current_" + params["saved_model_filename"]
    else:
        filename = "best_" + params["saved_model_filename"]

    save_path = os.path.join(params["checkpoint_dir"], filename)
    if os.path.exists(save_path):
        os.remove(save_path)
    print("saving model_guided to path:", save_path)
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_performance': best_performance,
        'current_performance': current_performance,
    }, save_path, _use_new_zipfile_serialization=False)


def train_loop(model, optimizer, train_loader, test_loader, params=None):
    saved_model_path = os.path.join(params["checkpoint_dir"], "current_" + params["saved_model_filename"])

    resume = params["resume"]
    if resume and os.path.isfile(saved_model_path):
        data_dict = torch.load(saved_model_path)
        epoch_start = data_dict["epoch"]
        if type(model) == nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        optimizer.load_state_dict(data_dict["optimizer"])
        history = data_dict["history"]
        best_performance = data_dict["best_performance"]
        if "current_performance" in list(data_dict.keys()):
            current_performance = data_dict["current_performance"]
            print("current_performance is:", current_performance)
        print("best_performance is:", best_performance)
    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"Cube_Avg_Loss": np.inf, "Voxel_Avg_Diff": np.inf}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

    loss_ave = 0
    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)

        model.train()
        values = {
            "non_padding_locations info": 0,
            "voxel_differ_avg (HU) info": 0,
            "non_padding_locations query": 0,
            "voxel_differ_avg (HU) query": 0,

            "num_voxel_depth > 4 info": 0,
            "num_voxel_depth > 4 query": 0,
            "voxel_differ_avg (HU) encoding_depth > 4, info": 0,
            "voxel_differ_avg (HU) encoding_depth > 4, query": 0
        }
        for i, batch_sample in enumerate(train_loader):

            list_query_gt_sequence = batch_sample["list_query_gt_sequence"]
            list_query_sequence = []

            num_cubes_query = 0
            for query_gt_sequence in list_query_gt_sequence:
                query_loc_sequence = []
                for item in query_gt_sequence:
                    query_loc_sequence.append(item['location_offset'])
                if len(query_loc_sequence) > num_cubes_query:
                    num_cubes_query = len(query_loc_sequence)
                list_query_sequence.append(query_loc_sequence)

            list_information_sequence = batch_sample["list_sample_sequence"]

            """
            # to see how overlap is predicted and how new loc is predicted
            num_cubes_info = 0
            list_info_loc_sequence = []
            for information_sequence in list_sample_sequence:
                info_loc_sequence = []
                for item in information_sequence:
                    info_loc_sequence.append(item['location_offset'])
                if len(information_sequence) > num_cubes_info:
                    num_cubes_info = len(information_sequence)
                list_info_loc_sequence.append(info_loc_sequence)
            

            if i == 0:
                print("number query cubes:", num_cubes_query)
                print("number information cubes:", num_cubes_info)
            
            # 1 for query loc that occurred in information
            stats_array = np.zeros([len(list_query_sequence), num_cubes_query], 'float32')
            """

            batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape = \
                utlis.prepare_tensors_3d_mae(list_information_sequence, list_query_sequence, params["embed_dim"],
                                             params["decoder_embed_dim"], params["given_dim"], device='cuda:0')

            batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query = \
                utlis.form_flatten_mask_mae(batch_sample, device='cuda:0')

            tensor_gt, tensor_penalty = loss_function.form_tensors_tissue_wise_v3(
                batch_sample, global_penalty_weight=(10, 0.1))

            tensor_gt = tensor_gt.to('cuda:1')
            tensor_penalty = tensor_penalty.to('cuda:1')

            prediction_vectors = model(batch_tensor, pos_embed_tensor, given_vector, query_vector,
                                       batch_mask_info, batch_mask_query)

            del batch_tensor, pos_embed_tensor, given_vector, query_vector

            tensor_predict = utlis.post_process_to_tensor(prediction_vectors, params["cube_size"]).to('cuda:1')

            flatten_dim = params["cube_size"][0] * params["cube_size"][1] * params["cube_size"][2]
            flatten_batch_depth_info = flatten_batch(batch_depth_info, flatten_dim)
            flatten_batch_depth_query = flatten_batch(batch_depth_query, flatten_dim)
            tensor_depth = utlis.post_process_to_tensor(
                torch.cat((flatten_batch_depth_info, flatten_batch_depth_query), dim=1), params["cube_size"])

            if i == 0:
                print("size for batch_depth_info", batch_depth_info.size())
                print("size for batch_depth_query", batch_depth_query.size())
                print("size for flatten_batch_depth_info", flatten_batch_depth_info.size())
                print("size for flatten_batch_depth_query", flatten_batch_depth_query.size())
                print("size for tensor_gt", tensor_gt.size())
                print("size for tensor_depth", tensor_depth.size())

            del batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query

            temp_depth_mask_array = np.array(tensor_depth.cpu().numpy() >= 4, 'float32')
            tensor_mask_depth_greater_4 = torch.FloatTensor(temp_depth_mask_array).cuda('cuda:1')

            # apply this, let it converge, then enhance >4 by 10 times
            loss = loss_function.weighted_absolute_difference_loss_tissue_wise(
                tensor_predict, tensor_gt, tensor_penalty, tissue_weight_tuple=params["tissue_weight_tuple"])

            # loss = loss_function.weighted_absolute_difference_loss_tissue_wise(
            #     10 * tensor_predict * tensor_mask_depth_greater_4 +
            #     tensor_predict, 10 * tensor_gt * tensor_mask_depth_greater_4 + tensor_gt, tensor_penalty,
            #     tissue_weight_tuple=params["tissue_weight_tuple"])

            loss_ave += loss.float().cpu().data

            loss = loss / params["accumulate_step"]
            loss.backward()

            if (i + 1) % params["accumulate_step"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss))

            ##########################
            # statics on training set
            ##########################

            tensor_predict = tensor_predict.detach().cpu().numpy()
            tensor_gt = tensor_gt.detach().cpu().numpy()
            tensor_mask_depth_greater_4 = tensor_mask_depth_greater_4.detach().cpu().numpy()

            num_information_cubes = 0
            batch_size = len(list_information_sequence)
            for num_batch in range(batch_size):
                if len(list_information_sequence[num_batch]) > num_information_cubes:
                    num_information_cubes = len(list_information_sequence[num_batch])

            tensor_gt_info = tensor_gt[:, 0: num_information_cubes]
            tensor_gt_query = tensor_gt[:, num_information_cubes::]
            tensor_predict_info = tensor_predict[:, 0: num_information_cubes]
            tensor_predict_query = tensor_predict[:, num_information_cubes::]

            mask_higher_zero_info = np.array(tensor_gt_info > 0, 'float32')
            mask_lower_zero_info = np.array(tensor_gt_info < 0, 'float32')
            mask_non_zero_info = mask_higher_zero_info + mask_lower_zero_info

            mask_higher_zero_query = np.array(tensor_gt_query > 0, 'float32')
            mask_lower_zero_query = np.array(tensor_gt_query < 0, 'float32')
            mask_non_zero_query = mask_higher_zero_query + mask_lower_zero_query

            mask_depth_greater_4_info = tensor_mask_depth_greater_4[:, 0: num_information_cubes, :]
            mask_depth_greater_4_query = tensor_mask_depth_greater_4[:, num_information_cubes::, :]

            values["voxel_differ_avg (HU) info"] += \
                np.sum(np.abs((tensor_gt_info - tensor_predict_info) * mask_non_zero_info))
            values["voxel_differ_avg (HU) query"] += \
                np.sum(np.abs((tensor_gt_query - tensor_predict_query) * mask_non_zero_query))

            values["voxel_differ_avg (HU) encoding_depth > 4, info"] += \
                np.sum(np.abs((tensor_gt_info - tensor_predict_info) * mask_depth_greater_4_info))
            values["voxel_differ_avg (HU) encoding_depth > 4, query"] += \
                np.sum(np.abs((tensor_gt_query - tensor_predict_query) * mask_depth_greater_4_query))

            num_voxel_depth_greater_4_info = np.sum(tensor_mask_depth_greater_4[:, 0: num_information_cubes, :])
            num_voxel_depth_greater_4_query = np.sum(tensor_mask_depth_greater_4[:, num_information_cubes::, :])
            values["num_voxel_depth > 4 info"] += num_voxel_depth_greater_4_info
            values["num_voxel_depth > 4 query"] += num_voxel_depth_greater_4_query

            for sequence in list_query_sequence:
                values["non_padding_locations query"] += len(sequence)
            for sequence in list_information_sequence:
                values["non_padding_locations info"] += len(sequence)

            del tensor_predict, tensor_gt, tensor_mask_depth_greater_4, mask_higher_zero_info, mask_lower_zero_info
            del mask_higher_zero_query, mask_lower_zero_query, mask_non_zero_info, mask_non_zero_query

        x, y, z = params["cube_size"]
        values["voxel_differ_avg (HU) info"] = \
            values["voxel_differ_avg (HU) info"] / values["non_padding_locations info"] / x / y / z * 1600
        values["voxel_differ_avg (HU) query"] = \
            values["voxel_differ_avg (HU) query"] / values["non_padding_locations query"] / x / y / z * 1600
        values["voxel_differ_avg (HU) encoding_depth > 4, info"] = \
            values["voxel_differ_avg (HU) encoding_depth > 4, info"] / values["num_voxel_depth > 4 info"] * 1600
        values["voxel_differ_avg (HU) encoding_depth > 4, query"] = \
            values["voxel_differ_avg (HU) encoding_depth > 4, query"] / values["num_voxel_depth > 4 query"] * 1600

        print("static on training dataset is:")
        for key, value in values.items():
            print(key, value)

        loss_ave = loss_ave / len(train_loader) / params["batch_ct"] / params["batch_split"]
        print("loss average on each CT scan:", loss_ave)
        history["loss_average_on_each_scan"].append(loss_ave)

        print("\tEvaluating")
        eval_values_train = evaluate(model, test_loader, params)

        for k, v in eval_values_train.items():  # store the history
            history[k + "_train"].append(v)

        print("\tCube_Avg_Loss=%.4f, Voxel_Diff_Avg=%.4f (HU), Total_Pred=%.4f, None_Pad_Pred=%.4f"
              % (
                  eval_values_train["loss_cube_avg"], eval_values_train["voxel_differ_avg (HU)"],
                  eval_values_train["total_locations"],
                  eval_values_train["non_padding_locations"]))

        current_performance = {"Cube_Avg_Loss": eval_values_train["loss_cube_avg"],
                               "Voxel_Avg_Diff": eval_values_train["voxel_differ_avg (HU)"]}

        if eval_values_train["loss_cube_avg"] < best_performance["Cube_Avg_Loss"]:
            print("New best model_guided at cube_avg_loss:", eval_values_train["loss_cube_avg"])
            best_performance["Cube_Avg_Loss"] = eval_values_train["loss_cube_avg"]
            best_performance["Voxel_Avg_Diff"] = eval_values_train["voxel_differ_avg (HU)"]
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, False)

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, params, show=True):
    model.eval()
    with torch.no_grad():
        values = {
            "loss_cube_avg": 0,  # the average
            "non_padding_locations": 0,
            "total_locations": 0,
            "voxel_differ_avg (HU)": 0,  # the average difference between predicted and ground truth
        }
        x, y, z = params["cube_size"]
        for i, batch_sample in enumerate(test_loader):

            list_query_gt_sequence = batch_sample["list_query_gt_sequence"]
            list_query_sequence = []
            for query_gt_sequence in list_query_gt_sequence:
                query_loc_sequence = []
                for item in query_gt_sequence:
                    query_loc_sequence.append(item['location_offset'])
                list_query_sequence.append(query_loc_sequence)
            list_information_sequence = batch_sample["list_sample_sequence"]

            max_query_length = 0
            for sequence in list_query_sequence:
                values["non_padding_locations"] += len(sequence)
                if max_query_length < len(sequence):
                    max_query_length = len(sequence)

            values["total_locations"] += max_query_length * len(list_query_sequence)

            tensor_gt, tensor_penalty = loss_function.form_tensors_tissue_wise_v3(
                batch_sample, global_penalty_weight=(10, 0.1), training=False)

            batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape = \
                utlis.prepare_tensors_3d_mae(list_information_sequence, list_query_sequence, params["embed_dim"],
                                             params["decoder_embed_dim"], params["given_dim"], device='cuda:1')

            batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query = \
                utlis.form_flatten_mask_mae(batch_sample, device='cuda:1')

            tensor_gt = tensor_gt.to('cuda:1')

            ############################################################################################
            # the dataset may be noisy, so some airway penalty is exploded, here set all_file voxels to be default equal,
            # you can change the global penalty in training control: params["tissue_weight_tuple"]
            tensor_penalty = torch.ones(tensor_penalty.size()).float()
            ############################################################################################

            tensor_penalty = tensor_penalty.to('cuda:1')

            prediction_vectors = model(batch_tensor, pos_embed_tensor, given_vector, query_vector,
                                       batch_mask_info, batch_mask_query)

            del batch_tensor, pos_embed_tensor, given_vector, query_vector

            del batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query

            tensor_predict = utlis.post_process_to_tensor(prediction_vectors, params["cube_size"]).to('cuda:1')

            mask_higher_zero = (tensor_gt > 0).float()
            mask_lower_zero = (tensor_gt < 0).float()
            mask_non_zero = mask_higher_zero + mask_lower_zero

            values["voxel_differ_avg (HU)"] += torch.sum(torch.abs((tensor_gt - tensor_predict) * mask_non_zero))

            loss_batch = loss_function.weighted_absolute_difference_loss_tissue_wise(
                tensor_predict, tensor_gt, tensor_penalty, mean=True, tissue_weight_tuple=params["tissue_weight_tuple"])

            del tensor_predict, tensor_gt, tensor_penalty, mask_non_zero, mask_higher_zero, mask_lower_zero

            values["loss_cube_avg"] += loss_batch

        if params["penalty_padding_cube"] is None:
            values["loss_cube_avg"] = values["loss_cube_avg"] / values["non_padding_locations"]
        else:
            values["loss_cube_avg"] = values["loss_cube_avg"] / values["total_locations"]

        values["voxel_differ_avg (HU)"] = \
            values['voxel_differ_avg (HU)'] / values["non_padding_locations"] / x / y / z * 1600

        if show:
            print(values)

        return values


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    train_dataset = dataset_and_loader.WeightedChestDatasetTransformer(
        params["train_data_dir"],
        mode='train',
        test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"]
    )

    test_dataset = dataset_and_loader.WeightedChestDatasetTransformer(
        params["train_data_dir"],
        mode='test',
        test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"]
    )

    print("train:", params["train_data_dir"], len(train_dataset))
    print("there are:", len(train_dataset), "training ct scans")
    print("there are:", len(test_dataset), "testing ct scans")

    train_loader = dataset_and_loader.DataLoaderPEIte(train_dataset, params["batch_ct"], params["batch_split"],
                                                      params["ratio_mask_input"], params["ratio_predict"],
                                                      params["input_output_overlap"], shuffle=True,
                                                      num_workers=params["num_workers"], drop_last=params["drop_last"],
                                                      pin_memory=params["pin_memory"], show=False,
                                                      min_parallel_ct=params["min_parallel_ct"],
                                                      mode='train',
                                                      num_prepared_dataset_train=params["num_prepared_dataset_train"],
                                                      num_prepared_dataset_test=params["num_prepared_dataset_test"],
                                                      reuse_count=params["reuse_count"])

    test_loader = dataset_and_loader.DataLoaderPEIte(test_dataset, params["batch_ct_test"], params["batch_split"],
                                                     params["ratio_mask_input"], params["ratio_predict"],
                                                     params["input_output_overlap"], shuffle=False,
                                                     num_workers=params["num_workers"], drop_last=False,
                                                     pin_memory=params["pin_memory"], show=False,
                                                     min_parallel_ct=params["min_parallel_ct"],
                                                     mode='test',
                                                     num_prepared_dataset_train=params["num_prepared_dataset_train"],
                                                     num_prepared_dataset_test=params["num_prepared_dataset_test"],
                                                     reuse_count=params["reuse_count"])

    model = model_mae.MAEGuidedSkipConnect(params["cube_size"], params["in_channel"], params["embed_dim"],
                                           params["given_dim"], params["encoder_depth"],
                                           params["encoder_heads"],
                                           params["decoder_embed_dim"], params["decoder_depth"],
                                           params["decoder_heads"], params["mlp_ratio"], show=True,
                                           extra_decoder_depth=params["extra_decoder_depth"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    train_loop(model, optimizer, train_loader, test_loader, params)
