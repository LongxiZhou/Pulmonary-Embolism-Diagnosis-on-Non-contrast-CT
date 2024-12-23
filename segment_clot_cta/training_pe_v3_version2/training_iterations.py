"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import pulmonary_embolism_v3.models.model_transformer as model_transformer
import pulmonary_embolism_v3.training.loss_function as loss_function
import segment_clot_cta.training_pe_v3_version2.dataset_and_dataloader as dataset_and_loader
from pulmonary_embolism_v3.utlis.phase_control_and_sample_process import \
    put_arrays_on_device, OutlierLossDetect, TrainingPhaseControl


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
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    clot_dataset = dataset_and_loader.ClotDataset(params["top_dict_clot_pickle"], mode=params['mode'])

    original_sample_dataset_train = dataset_and_loader.SampleDataset(
        params["sample_dir_list"], 'train', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"],
        shuffle_path_list=params["shuffle_path_list"])
    original_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    annotated_sample_dataset_train = dataset_and_loader.SampleDataset(
        params["sample_dir_list_with_gt"], 'train', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"],
        shuffle_path_list=params["shuffle_path_list"])
    annotated_sample_dataset_test = dataset_and_loader.SampleDataset(
        params["sample_dir_list_with_gt"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"],
        shuffle_path_list=params["shuffle_path_list"])

    train_loader = dataset_and_loader.DataLoaderSimulatedClot(
        clot_dataset, original_sample_dataset_train, params["batch_size"], shuffle=True,
        num_workers=params["num_workers"], mode="train", clot_volume_range=params["clot_volume_range"],
        min_clot=params["min_clot"], num_clot_each_sample_range=params["num_clot_each_sample_range"],
        augment=params["augment"], embed_dim=params["embed_dim"], trace_clot=params["trace_clot"], roi=params["roi"],
        global_bias_range=params["global_bias_range"], annotated_sample_dataset=annotated_sample_dataset_train,
        relative_frequency_simulate_gt=params["relative_frequency_simulate_gt"],
        relative_frequency_v1_v2=params["relative_frequency_v1_v2"])

    test_loader = dataset_and_loader.DataLoaderSimulatedClot(
        clot_dataset, original_sample_dataset_test, params["batch_size_test"], shuffle=False,
        num_workers=params["num_workers"], mode="test", clot_volume_range=params["clot_volume_range"],
        min_clot=params["min_clot"], num_clot_each_sample_range=params["num_clot_each_sample_range"],
        augment=params["augment"], num_prepared_dataset_test=params["num_prepared_dataset_test"],
        embed_dim=params["embed_dim"], trace_clot=params["trace_clot"], roi=params["roi"],
        global_bias_range=params["global_bias_range"], annotated_sample_dataset=annotated_sample_dataset_test,
        relative_frequency_simulate_gt=params["relative_frequency_simulate_gt"],
        relative_frequency_v1_v2=params["relative_frequency_v1_v2"])

    print("there are:", len(original_sample_dataset_train), "training ct scans")
    print("there are:", len(original_sample_dataset_test), "testing ct scans")

    train_loader.update_clot_simulation_parameter_v1(global_bias_range=params["global_bias_range"])
    test_loader.update_clot_simulation_parameter_v1(global_bias_range=params["global_bias_range"])

    train_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                     params["value_increase"], params["voxel_variance"])
    test_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                    params["value_increase"], params["voxel_variance"])

    train_loop(model, optimizer, train_loader, test_loader, params)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True,
                    training_phase_control=None, special_name=None, outlier_loss_detect=None, global_bias_range=None,
                    value_increase=None):
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
        'global_bias_range': global_bias_range,
        'value_increase': value_increase,
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
            outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)

        if params["global_bias_range"] is not None:
            global_bias_range = list(params["global_bias_range"])
            print("Using given global bias range:", global_bias_range)
        else:
            global_bias_range = data_dict['global_bias_range']

        if params["value_increase"] is not None:
            value_increase = list(params["value_increase"])
        else:
            value_increase = data_dict['value_increase']

    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}
        training_phase_control = TrainingPhaseControl(params)
        outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)
        global_bias_range = list(params["global_bias_range"])
        value_increase = list(params["value_increase"])

    if params["reset_best_performance"]:
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))
    print("global bias range:", global_bias_range)
    print("value increase:", value_increase)

    train_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                     value_increase, params["voxel_variance"])
    test_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                    value_increase, params["voxel_variance"])

    train_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)
    test_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)

    softmax_layer = torch.nn.Softmax(dim=1)
    training_phase_control.flip_remaining = params["flip_remaining"]
    training_phase_control.flip_recall = params["flip_recall"]
    training_phase_control.flip_precision = params["flip_precision"]
    training_phase_control.base_relative = params["base_relative"]
    training_phase_control.max_performance_recall = params["max_performance_recall"]
    training_phase_control.max_performance_precision = params["max_performance_precision"]
    training_phase_control.warm_up_epochs = params["warm_up_epochs"]

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
            # batch_sample is (array_packages, list_whether_important)

            if model_failed:
                train_loader.force_stop_iteration()
                continue

            array_packages = batch_sample[0]
            list_sample_attention = []
            for whether_important in batch_sample[1]:  # list_whether_important
                if whether_important:
                    list_sample_attention.append(params["weight_for_important"])
                else:
                    list_sample_attention.append(1.)

            batch_tensor, pos_embed_tensor, given_vector, flatten_roi_region, cube_shape, clot_gt_tensor, \
                penalty_weight_tensor = put_arrays_on_device(
                    array_packages, device='cuda:0', training_phase=True, penalty_normalize_func=None,
                    trace_clot=train_loader.trace_clot)

            segmentation_before_softmax = model(
                batch_tensor, pos_embed_tensor, given_vector, flatten_roi_region)
            # [B, 2, N, flatten_dim]

            loss = loss_function.weighted_cross_entropy_loss(segmentation_before_softmax, clot_gt_tensor, class_balance,
                                                             list_sample_attention, penalty_weight_tensor)
            if i % 10 == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss))

            float_loss = loss.detach().float().cpu().data

            loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss
            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                del loss, segmentation_before_softmax, clot_gt_tensor
                del batch_tensor, pos_embed_tensor, given_vector, flatten_roi_region, cube_shape

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
                print("size for flatten_blood_region", flatten_roi_region.size())
                print("size for clot_gt_tensor", clot_gt_tensor.size())
                print("size for segmentation_before_softmax", segmentation_before_softmax.size())
                print("initial class balance:", class_balance)
                print("list_clot_attention:", list_sample_attention)

            del batch_tensor, pos_embed_tensor, given_vector, flatten_roi_region
            del segmentation_before_softmax, clot_mask_gt

        if model_failed:
            print("failure model, roll back to back up version")

            backup_model_path = os.path.join(params["checkpoint_dir"], "backup_" + params["saved_model_filename"])

            data_dict = torch.load(backup_model_path)
            if type(model) == nn.DataParallel:
                model.module.load_state_dict(data_dict["state_dict"])
            else:
                model.load_state_dict(data_dict["state_dict"])

            training_phase_control = data_dict['phase_control']
            outlier_loss_detect = data_dict['outlier_loss_detect']
            optimizer.load_state_dict(data_dict["optimizer"])

            global_bias_range = data_dict['global_bias_range']
            print("back up version has global_bias_range:", global_bias_range)

            value_increase = data_dict['value_increase']
            print("back up version has value increase:", value_increase)

            train_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                             value_increase, params["voxel_variance"])
            test_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                            value_increase, params["voxel_variance"])
            train_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)
            test_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)

            model_failed = False  # rolled back
            continue

        if accumulative_step == 0:
            raise EnvironmentError("iteration cannot be started")

        recall = num_true_positive / (total_clot_voxel + 0.0001)
        precision = num_true_positive / (num_true_positive + num_false_positive + 0.0001)

        if recall == 0 or np.isnan(recall) or precision == 0 or np.isnan(precision):
            if training_phase_control.current_phase is not 'warm_up':
                model_failed = True

        if recall <= 0 or precision <= 0:
            dice = 0
        else:
            dice = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / accumulative_step / params["batch_size"]
        print("\nloss average on each CT scan training:", loss_ave)
        print("recall on training:", recall)
        print("precision on training:", precision)
        print("dice on training:", dice, '\n')
        history["loss_average_on_each_scan_training"].append(loss_ave)
        history["recall_for_each_training_epoch"].append(recall)
        history["precision_for_each_training_epoch"].append(precision)

        print("\tEvaluating")

        loss_ave_test, recall_test, precision_test, dice_test = \
            evaluate(model, test_loader, params, training_phase_control, history)

        if recall_test == 0 or np.isnan(recall_test) or precision_test == 0 or np.isnan(precision_test):
            if training_phase_control.current_phase is not 'warm_up':
                model_failed = True

        if dice == 0 or np.isnan(dice) or dice_test == 0 or np.isnan(dice_test):
            if training_phase_control.current_phase is not 'warm_up':
                model_failed = True

        current_performance = {"loss_ave_train": loss_ave, "loss_ave_test": loss_ave_test,
                               "recall_train": recall, "recall_test": recall_test,
                               "precision_train": precision, "precision_test": precision_test,
                               "dice_train": dice, "dice_test": dice_test,
                               "relative_false_positive_penalty": relative_false_positive_penalty}

        if current_performance["dice_test"] < best_performance["dice_test"]:
            if current_performance["dice_test"] < params['min_dice']:
                if training_phase_control.current_phase is not 'warm_up':
                    model_failed = True

        if training_phase_control.changed_phase_in_last_epoch:
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                            training_phase_control=training_phase_control,
                            special_name="gb_" + str(global_bias_range[0])[0: 8] + '_dice_' +
                                         str(dice_test)[0: 5] + '_' + training_phase_control.previous_phase,
                            outlier_loss_detect=outlier_loss_detect, global_bias_range=global_bias_range,
                            value_increase=value_increase)

        if current_performance["dice_test"] > best_performance["dice_test"]:
            print("\nNew best model_guided at dice test:", current_performance["dice_test"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect,
                            global_bias_range=global_bias_range, value_increase=value_increase)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect,
                        global_bias_range=global_bias_range, value_increase=value_increase)

        flip_remaining = training_phase_control.flip_remaining

        if flip_remaining == 1:
            print("model finished one flip")

            if dice_test > params['min_dice_backup']:
                print("updating backup model at dice test:", dice_test)
                save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                                best=False,
                                training_phase_control=training_phase_control,
                                special_name="backup",
                                outlier_loss_detect=outlier_loss_detect,
                                global_bias_range=global_bias_range,
                                value_increase=value_increase)
            if dice_test < params['min_dice_at_flip']:
                print("model failed at dice test on flip:", dice_test)
                model_failed = True  # detect failure in the evaluation
            if dice_test < (best_performance["dice_test"] - params['min_dice_less_than_best']):
                print("model failed at dice test on flip:", dice_test)
                model_failed = True  # detect failure in the evaluation

            training_phase_control.flip_remaining += 1

            if params["difficulty"] == "stable":
                continue

            if params["difficulty"] == "increase":
                global_bias_range[0] = global_bias_range[0] / 1.1
                global_bias_range[1] = global_bias_range[1] / 1.1
                value_increase[0] = value_increase[0] / 1.1
                value_increase[1] = value_increase[1] / 1.1
            if params["difficulty"] == "decrease":
                global_bias_range[0] = global_bias_range[0] * 1.1
                global_bias_range[1] = global_bias_range[1] * 1.1
                value_increase[0] = value_increase[0] * 1.1
                value_increase[1] = value_increase[1] * 1.1

            if abs(value_increase[0]) < 0.025:
                value_increase = [-0.025, -0.125]  # -40 HU, -200 HU

            if abs(global_bias_range[0]) <= 0.0003:  # 0.48 HU
                if params["difficulty"] == "increase":
                    params["difficulty"] = "stable"
                    global_bias_range = [0., 0.]

            train_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)
            test_loader.update_clot_simulation_parameter_v1(global_bias_range=global_bias_range)
            train_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                             value_increase, params["voxel_variance"])
            test_loader.update_clot_simulation_parameter_v2(params["power_range"], params["add_base_range"],
                                                            value_increase, params["voxel_variance"])

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, params, training_phase_control, history):
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
            array_packages = batch_sample[0]
            list_sample_attention = []
            for whether_important in batch_sample[1]:  # list_whether_important
                if whether_important:
                    list_sample_attention.append(params["weight_for_important"])
                else:
                    list_sample_attention.append(1.)

            batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region, cube_shape, clot_gt_tensor, \
                penalty_weight_tensor = put_arrays_on_device(
                    array_packages, device='cuda:0', training_phase=True, penalty_normalize_func=None,
                    trace_clot=test_loader.trace_clot)

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

        loss_ave = loss_ave / len(test_loader) / params["batch_size_test"]
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
