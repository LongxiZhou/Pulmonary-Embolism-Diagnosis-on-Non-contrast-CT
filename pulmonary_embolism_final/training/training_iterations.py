"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import pulmonary_embolism_final.models.model_transformer as model_transformer
import pulmonary_embolism_final.training.loss_function as loss_function
import pulmonary_embolism_final.training.dataset_and_loader as dataset_and_loader
from pulmonary_embolism_final.utlis.phase_control_and_outlier_loss_detect import \
    OutlierLossDetect, TrainingPhaseControl
import pulmonary_embolism_final.utlis.sample_to_tensor_with_annotation as sample_to_tensor_with_annotation
import pulmonary_embolism_final.utlis.sample_to_tensor_simulate_clot as sample_to_tensor_simulate_clot
from functools import partial


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    model = model_transformer.GuidedWithBranch(
        params["cube_size"], params["in_channel"], params["cnn_features"], params["given_features"],
        params["embed_dim"], params["num_heads"], params["encoding_depth"], params["interaction_depth"],
        params["decoding_depth"], params["segmentation_depth"], params["mlp_ratio"]
    )

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=params["device_ids"])
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    clot_dataset = dataset_and_loader.ClotDataset(params["top_dict_clot_pickle"], mode=params['mode'])
    non_pe_sample_dataset_train = dataset_and_loader.SampleDataset(
        params["sample_dir_list_non_pe"], 'train', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])
    non_pe_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list_non_pe"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    pe_sample_dataset_train = dataset_and_loader.SampleDataset(
        params["sample_dir_list_pe"], 'train', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])
    pe_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list_pe"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    train_loader_non_pe = dataset_and_loader.DataLoaderSimulatedClot(
        clot_dataset, non_pe_sample_dataset_train, params["batch_size_simu"], shuffle=True,
        num_workers=params["num_workers_simu"], mode="train", clot_volume_range=params["clot_volume_range"],
        min_clot=params["min_clot"], num_clot_each_sample_range=params["num_clot_each_sample_range"],
        augment=params["augment_non_pe"], embed_dim=params["embed_dim"],
        sample_sequence_length=params["sample_sequence_length"])
    test_loader_non_pe = dataset_and_loader.DataLoaderSimulatedClot(
        clot_dataset, non_pe_sample_dataset_test, params["batch_size_simu"], shuffle=False,
        num_workers=params["num_workers_simu"], mode="test", clot_volume_range=params["clot_volume_range"],
        min_clot=params["min_clot"], num_clot_each_sample_range=params["num_clot_each_sample_range"],
        augment=params["augment_non_pe"], num_prepared_dataset_test=params["num_prepared_dataset_test"],
        embed_dim=params["embed_dim"], sample_sequence_length=params["sample_sequence_length"])

    train_loader_pe = dataset_and_loader.DataLoaderWithAnnotation(
        pe_sample_dataset_train, params["batch_size_with_gt"], num_workers=params["num_workers_with_gt"],
        augment=params["augment_pe"], random_select=False, shuffle=True, embed_dim=params["embed_dim"],
        sample_sequence_length=params["sample_sequence_length"])
    test_loader_pe = dataset_and_loader.DataLoaderWithAnnotation(
        pe_sample_dataset_test, params["batch_size_with_gt"], num_workers=params["num_workers_with_gt"],
        augment=params["augment_pe"], random_select=False, shuffle=True, embed_dim=params["embed_dim"],
        sample_sequence_length=params["sample_sequence_length"])

    print("there are:", len(non_pe_sample_dataset_train), "training ct scans")
    print("there are:", len(non_pe_sample_dataset_test), "testing ct scans")

    train_loader_non_pe.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                         params["value_increase"], params["voxel_variance"])
    test_loader_non_pe.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                        params["value_increase"], params["voxel_variance"])

    train_loop(model, optimizer, train_loader_non_pe, test_loader_non_pe,
               train_loader_pe, test_loader_pe, params)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True,
                    training_phase_control=None, special_name=None, outlier_loss_detect=None, value_increase=None):
    if not best:  # this means we store the current model_guided
        filename = "current_" + params["saved_model_filename"]
    else:
        filename = "best_" + params["saved_model_filename"]
    if params["mode"] == 'temp':
        filename = "temp_" + filename
    if special_name is not None:
        filename = special_name + "_" + params["saved_model_filename"]

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
        'phase_control': training_phase_control,
        'outlier_loss_detect': outlier_loss_detect,
        'value_increase': value_increase,
    }, save_path, _use_new_zipfile_serialization=False)


def train_loop(model, optimizer, train_loader, test_loader, train_loader_pe, test_loader_pe, params=None):
    saved_model_path = os.path.join(params["checkpoint_dir"], "current_" + params["saved_model_filename"])

    penalty_normalize_func_with_gt = partial(sample_to_tensor_with_annotation.default_penalty_normalize_func,
                                             relative_ratio=params["relative_ratio_with_gt"])

    resume = params["resume"]
    if resume and os.path.isfile(saved_model_path):
        data_dict = torch.load(saved_model_path)
        epoch_start = data_dict["epoch"]
        if type(model) == nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        if params["reuse_optimizer"]:
            optimizer.load_state_dict(data_dict["optimizer"])
        history = data_dict["history"]
        best_performance = data_dict["best_performance"]
        if "current_performance" in list(data_dict.keys()):
            current_performance = data_dict["current_performance"]
            print("current_performance is:", current_performance)
        print("best_performance is:", best_performance)

        if params["reuse_phase_control"]:
            training_phase_control = data_dict['phase_control']
        else:
            training_phase_control = TrainingPhaseControl(params)

        outlier_loss_detect = data_dict['outlier_loss_detect']
        if outlier_loss_detect is None or params["reset_outlier_detect"]:
            outlier_loss_detect = OutlierLossDetect(max(30, int(3 * params["accumulate_step"])), 3, 3, 10)

        if params["value_increase"] is not None:
            value_increase = list(params["value_increase"])
            print("Using given value increase:", value_increase)
        else:
            value_increase = data_dict['value_increase']

    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}
        training_phase_control = TrainingPhaseControl(params)
        outlier_loss_detect = OutlierLossDetect(max(30, int(3 * params["accumulate_step"])), 3, 3, 10)
        value_increase = list(params["value_increase"])

    if params["reset_best_performance"]:
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))
    print("value increase:", value_increase)

    train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                  value_increase, params["voxel_variance"])
    test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                 value_increase, params["voxel_variance"])

    softmax_layer = torch.nn.Softmax(dim=1)
    training_phase_control.flip_remaining = params["flip_remaining"]
    training_phase_control.flip_recall = params["flip_recall"]
    training_phase_control.flip_precision = params["flip_precision"]
    training_phase_control.base_relative = params["base_relative"]
    training_phase_control.max_performance_recall = params["max_performance_recall"]
    training_phase_control.max_performance_precision = params["max_performance_precision"]

    print("flip_recall:", training_phase_control.flip_recall, "flip_precision:", training_phase_control.flip_precision,
          "base_relative:", training_phase_control.base_relative,
          "max_performance_recall:", training_phase_control.max_performance_recall,
          "max_performance_precision:", training_phase_control.max_performance_precision)

    model_failed = False

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)

        loss_ave = 0
        total_clot_voxel = 0
        num_true_positive = 0
        num_false_positive = 0

        accumulative_step = 0

        relative_false_positive_penalty = training_phase_control.relative_false_positive_penalty
        # higher means model give less false positives, at the expense of more false negative
        class_balance = [relative_false_positive_penalty, 100 / relative_false_positive_penalty]
        print("class balance:", class_balance)
        training_phase_control.show_status()

        model.train()
        for i, batch_sample in enumerate(train_loader):
            # batch_sample is (array_packages, list_importance_score)

            if model_failed:
                train_loader_pe.clear_sub_process_queue()
                # stop iteration will fail to join sub process if send data exceed 64 KB
                continue

            array_packages_a, list_sample_attention_a = batch_sample
            tensors_a = sample_to_tensor_simulate_clot.put_arrays_on_device_simu_clot(
                array_packages_a, device='cuda:0', training_phase=True, penalty_normalize_func=None)

            if i + train_loader_pe.num_workers < len(train_loader):
                need_new_process = True
            else:
                need_new_process = False

            array_packages_b, list_sample_attention_b = \
                train_loader_pe.extract_data_from_sub_process_and_start_a_new(need_new_process)
            tensors_b = sample_to_tensor_with_annotation.put_arrays_on_device_with_gt(
                array_packages_b, device='cuda:0', training_phase=True,
                penalty_normalize_func=penalty_normalize_func_with_gt)

            if i + 1 == len(train_loader):  # stop iteration will fail to join sub process if send data exceed 64 KB
                train_loader_pe.clear_sub_process_queue()

            (batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region,
             cube_shape, clot_gt_tensor, penalty_weight_tensor), list_sample_attention = \
                dataset_and_loader.merge_tensor_packages(
                    (tensors_a, list_sample_attention_a), (tensors_b, list_sample_attention_b))

            segmentation_before_softmax = model(
                batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region)
            # [B, 2, N, flatten_dim]

            loss = loss_function.weighted_cross_entropy_loss(segmentation_before_softmax, clot_gt_tensor, class_balance,
                                                             list_sample_attention, penalty_weight_tensor)
            if i % 10 == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss))

            float_loss = loss.detach().float().cpu().data

            loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss
            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                del loss, segmentation_before_softmax, clot_gt_tensor
                del batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region, cube_shape

                model_failed = True  # detect failure inside epoch
                continue

            if not loss_status:  # an outlier is detected
                std_in_queue, ave_in_queue = outlier_loss_detect.get_std_and_ave_in_queue()
                loss = loss / abs(float_loss - ave_in_queue) * std_in_queue  # reduce the weight for the loss

            accumulative_step += 1

            loss_ave += float_loss
            loss = loss / params["accumulate_step"]
            loss.backward()
            if (accumulative_step + 1) % params["accumulate_step"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            segmentation_before_softmax = segmentation_before_softmax.detach()

            segment_probability_clot = softmax_layer(segmentation_before_softmax).cpu().numpy()[:, 1, :, :]
            segment_mask_clot = np.array(segment_probability_clot > 0.5, 'float32')
            clot_mask_gt = clot_gt_tensor.detach().cpu().numpy()[:, 1, :, :]
            # [B, N, flatten_dim]
            if not np.min(clot_mask_gt) == 0 and np.max(clot_mask_gt) == 1:
                print("range for clot_mask_gt:", np.min(clot_mask_gt), np.max(clot_mask_gt))

            overlap_count_batch = np.sum(clot_mask_gt * segment_mask_clot)

            num_true_positive += overlap_count_batch
            total_clot_voxel += np.sum(clot_mask_gt)
            num_false_positive += np.sum(segment_mask_clot) - overlap_count_batch

            if i == 0 and epoch == 0:
                print("size for batch_tensor", batch_tensor.size())
                print("size for pos_embed_tensor", pos_embed_tensor.size())
                print("size for flatten_blood_region", flatten_blood_region.size())
                print("size for clot_gt_tensor", clot_gt_tensor.size())
                print("size for segmentation_before_softmax", segmentation_before_softmax.size())
                print("initial class balance:", class_balance)
                print("list_clot_attention:", list_sample_attention)

            del batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region
            del segmentation_before_softmax, clot_mask_gt

        if model_failed:
            print("failure model, roll back to back up version")

            backup_model_path = os.path.join(params["checkpoint_dir"], "backup_" + params["saved_model_filename"])

            data_dict = torch.load(backup_model_path)
            if type(model) == nn.DataParallel:
                model.module.load_state_dict(data_dict["state_dict"])
            else:
                model.load_state_dict(data_dict["state_dict"])

            optimizer.load_state_dict(data_dict["optimizer"])

            training_phase_control = data_dict['phase_control']
            outlier_loss_detect = data_dict['outlier_loss_detect']
            value_increase = data_dict['value_increase']

            print("back up version has value increase:", value_increase)

            train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                          value_increase, params["voxel_variance"])
            test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                         value_increase, params["voxel_variance"])

            model_failed = False  # rolled back
            continue

        recall = num_true_positive / total_clot_voxel
        precision = num_true_positive / (num_true_positive + num_false_positive)

        if recall <= 0 or precision <= 0:
            dice = 0
        else:
            dice = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / accumulative_step / (params["batch_size_simu"] + params["batch_size_with_gt"])
        print("\nloss average on each CT scan training:", loss_ave)
        print("recall on training:", recall)
        print("precision on training:", precision)
        print("dice on training:", dice, '\n')
        history["loss_average_on_each_scan_training"].append(loss_ave)
        history["recall_for_each_training_epoch"].append(recall)
        history["precision_for_each_training_epoch"].append(precision)

        print("\tEvaluating")

        loss_ave_test, recall_test, precision_test, dice_test = \
            evaluate(model, test_loader, test_loader_pe, params, training_phase_control, history)

        current_performance = {"loss_ave_train": loss_ave, "loss_ave_test": loss_ave_test,
                               "recall_train": recall, "recall_test": recall_test,
                               "precision_train": precision, "precision_test": precision_test,
                               "dice_train": dice, "dice_test": dice_test,
                               "relative_false_positive_penalty": relative_false_positive_penalty}

        if training_phase_control.changed_phase_in_last_epoch:
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                            training_phase_control=training_phase_control,
                            special_name="vi_" + str(value_increase[0])[0: 5] + '_dice_' +
                                         str(dice_test)[0: 5] + '_' + training_phase_control.previous_phase,
                            outlier_loss_detect=outlier_loss_detect, value_increase=value_increase)

        if current_performance["dice_test"] > best_performance["dice_test"]:
            print("\nNew best model_guided at dice test:", current_performance["dice_test"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect,
                            value_increase=value_increase)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect,
                        value_increase=value_increase)

        flip_remaining = training_phase_control.flip_remaining
        if flip_remaining == 1:

            if params["difficulty"] == "stable":
                if not params["converge_to_final_phase"]:
                    training_phase_control.flip_remaining += 1
                continue

            if dice_test > 0.2:
                print("updating backup model at dice test:", dice_test)
                save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                                best=False,
                                training_phase_control=training_phase_control,
                                special_name="backup",
                                outlier_loss_detect=outlier_loss_detect,
                                value_increase=value_increase)
            if dice_test < 0.1:
                print("model failed at dice test:", dice_test)
                model_failed = True  # detect failure in the evaluation

            if value_increase[0] >= 0.5:
                assert not params["difficulty"] == "decrease"
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.75
            elif 0.1 <= value_increase[0] < 0.5:
                assert not params["difficulty"] == "decrease"
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.5
            elif 0.03 <= value_increase[0] < 0.1:
                assert not params["difficulty"] == "decrease"
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.25
            else:
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.1
                if params["difficulty"] == "decrease":
                    value_increase[0] = value_increase[0] * 1.1

            if value_increase[0] <= 0.01:
                if params["difficulty"] == "increase":
                    params["difficulty"] = "decrease"
            if value_increase[0] >= 0.02:
                if params["difficulty"] == "decrease":
                    params["difficulty"] = "increase"

            value_increase[1] = value_increase[0] * 5
            train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                          value_increase, params["voxel_variance"])
            test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                         value_increase, params["voxel_variance"])

            outlier_loss_detect.reset()
            training_phase_control.flip_remaining += 1

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, test_loader_pe, params, training_phase_control, history):
    penalty_normalize_func_with_gt = partial(sample_to_tensor_with_annotation.default_penalty_normalize_func,
                                             relative_ratio=params["relative_ratio_with_gt"])
    loss_ave = 0
    total_clot_voxel = 0
    num_true_positive = 0
    num_false_positive = 0
    relative_false_positive_penalty = training_phase_control.relative_false_positive_penalty
    class_balance = [relative_false_positive_penalty, 100 / relative_false_positive_penalty]
    softmax_layer = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(test_loader):
            array_packages_a, list_sample_attention_a = batch_sample
            tensors_a = sample_to_tensor_simulate_clot.put_arrays_on_device_simu_clot(
                array_packages_a, device='cuda:0', training_phase=True, penalty_normalize_func=None)

            if i + test_loader_pe.num_workers < len(test_loader):
                need_new_process = True
            else:
                need_new_process = False
            array_packages_b, list_sample_attention_b = \
                test_loader_pe.extract_data_from_sub_process_and_start_a_new(need_new_process)
            tensors_b = sample_to_tensor_with_annotation.put_arrays_on_device_with_gt(
                array_packages_b, device='cuda:0', training_phase=True,
                penalty_normalize_func=penalty_normalize_func_with_gt)

            if i + 1 == len(test_loader):  # stop iteration will fail to join sub process if send data exceed 64 KB
                test_loader_pe.clear_sub_process_queue()

            (batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region,
             cube_shape, clot_gt_tensor, penalty_weight_tensor), list_sample_attention = \
                dataset_and_loader.merge_tensor_packages(
                    (tensors_a, list_sample_attention_a), (tensors_b, list_sample_attention_b))

            segmentation_before_softmax = model(
                batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region)
            # [B, 2, N, flatten_dim]

            loss = loss_function.weighted_cross_entropy_loss(segmentation_before_softmax, clot_gt_tensor, class_balance,
                                                             list_sample_attention, penalty_weight_tensor)

            loss_ave += loss.detach().float().cpu().data

            segment_probability_clot = softmax_layer(segmentation_before_softmax.detach()).cpu().numpy()[:, 1, :, :]
            segment_mask_clot = np.array(segment_probability_clot > 0.5, 'float32')
            clot_mask_gt = clot_gt_tensor.detach().cpu().numpy()[:, 1, :, :]
            # [B, N, flatten_dim]

            overlap_count_batch = np.sum(clot_mask_gt * segment_mask_clot)

            num_true_positive += overlap_count_batch
            total_clot_voxel += np.sum(clot_mask_gt)
            num_false_positive += np.sum(segment_mask_clot) - overlap_count_batch

        recall = num_true_positive / total_clot_voxel
        precision = num_true_positive / (num_true_positive + num_false_positive)

        if recall <= 0 or precision <= 0:
            dice_test = 0
        else:
            dice_test = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / len(test_loader) / (params["batch_size_simu"] + params["batch_size_with_gt"])
        print("\nloss average on each CT scan testing:", loss_ave)
        print("recall on testing:", recall)
        print("precision on testing:", precision)
        print("dice_test:", dice_test, '\n')

        history["loss_average_on_each_scan_testing"].append(loss_ave)
        history["recall_for_each_testing_epoch"].append(recall)
        history["precision_for_each_testing_epoch"].append(precision)
        history["dice_test"].append(dice_test)

        training_phase_control.get_new_relative_false_positive_penalty(recall, precision)

        return loss_ave, recall, precision, dice_test
