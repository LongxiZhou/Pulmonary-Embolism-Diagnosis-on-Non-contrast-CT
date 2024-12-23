"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import random
import collections
import registration_pulmonary.models.model_need_landmark as main_model
import registration_pulmonary.training_v3.dataset_and_dataloader as dataset_and_loader
import registration_pulmonary.loss_functions.image_based_loss as image_based_loss
import registration_pulmonary.loss_functions.flow_based_loss as flow_based_loss
from registration_pulmonary.utlis.loss_and_phase_control import OutlierLossDetect, TrainingPhaseControlFlowRoughness
from registration_pulmonary.simulation_and_augmentation.process_data_augmentation import \
    apply_translate_augmentation, set_fixed_and_registered_as_the_same


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    image_size = (params["image_length"], params["image_length"], params["image_length"])

    model = main_model.RefineRegistrationFlow(
        image_size=image_size, num_channel=4, num_landmark=params["num_landmark"],
        depth_get_landmark=params["depth_get_landmark"], depth_refine_flow=params["depth_refine_flow"],
        inference_phase=False, split_positive_and_negative=params["split_positive_and_negative"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    original_sample_dataset_train = dataset_and_loader.OriginalSampleDataset(
        params["sample_dir_list"], 'train', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"])
    original_sample_dataset_test = dataset_and_loader.OriginalSampleDataset(
        params["sample_dir_list"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"])

    train_loader = dataset_and_loader.DataLoaderRegistration(
        original_sample_dataset_train, params["batch_size_train"], shuffle=True, mode='train',
        augment=params["augment"], drop_last=True,
        ratio_swap=params["ratio_swap"], ratio_non_to_non=params["ratio_non_to_non"])

    test_loader = dataset_and_loader.DataLoaderRegistration(
        original_sample_dataset_test, params["batch_size_test"], shuffle=False, mode='test',
        augment=params["augment"], drop_last=False,
        ratio_swap=params["ratio_swap"], ratio_non_to_non=params["ratio_non_to_non"])

    print("there are:", len(original_sample_dataset_train), "training ct pairs")
    print("there are:", len(original_sample_dataset_test), "testing ct pairs")

    train_loop(model, optimizer, train_loader, test_loader, params)


def testing(params, saved_model_path=None):
    if saved_model_path is None:
        saved_model_path = os.path.join(params["checkpoint_dir"], "current_" + params["saved_model_filename"])

    image_size = (params["image_length"], params["image_length"], params["image_length"])

    model = main_model.RefineRegistrationFlow(
        image_size=image_size, num_channel=4, num_landmark=params["num_landmark"],
        depth_get_landmark=params["depth_get_landmark"], depth_refine_flow=params["depth_refine_flow"],
        inference_phase=False, split_positive_and_negative=params["split_positive_and_negative"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])

    data_dict = torch.load(saved_model_path)
    if type(model) == nn.DataParallel:
        model.module.load_state_dict(data_dict["state_dict"])
    else:
        model.load_state_dict(data_dict["state_dict"])

    original_sample_dataset_test = dataset_and_loader.OriginalSampleDataset(
        params["sample_dir_list"], 'test', params["test_id"], sample_interval=params["sample_interval"],
        wrong_file_name=params["wrong_file_name"], important_file_name=params["important_file_name"])

    test_loader = dataset_and_loader.DataLoaderRegistration(
        original_sample_dataset_test, params["batch_size_test"], shuffle=False, mode='test',
        augment=params["augment"], drop_last=False,
        ratio_swap=params["ratio_swap"], ratio_non_to_non=params["ratio_non_to_non"])

    print("there are:", len(original_sample_dataset_test), "testing ct pairs")
    loss_tracker_test = TrackLossValue(
        batch_size=params["batch_size_test"], loss_values_dict_epoch=collections.defaultdict(list))
    evaluate(model, test_loader, params, None, loss_tracker_test)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True,
                    training_phase_control=None, special_name=None, outlier_loss_detect=None):
    if not best:  # this means we store the current model_guided
        filename = "current_" + params["saved_model_filename"]
    else:
        filename = "best_" + params["saved_model_filename"]
    if special_name is not None:
        filename = special_name + "_" + params["saved_model_filename"]

    save_path = os.path.join(params["checkpoint_dir"], filename)
    if os.path.exists(save_path):
        os.remove(save_path)
    print("saving model_guided to path:", save_path)

    history_train, history_test = history

    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history_train': history_train,
        'history_test': history_test,
        'best_performance': best_performance,
        'current_performance': current_performance,
        'phase_control': training_phase_control,
        'outlier_loss_detect': outlier_loss_detect,
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
        history_train, history_test = data_dict["history_train"], data_dict["history_test"]
        best_performance = data_dict["best_performance"]
        if "current_performance" in list(data_dict.keys()):
            current_performance = data_dict["current_performance"]
            print("current_performance is:", current_performance)
        print("best_performance is:", best_performance)

        if params["reuse_phase_control"]:
            training_phase_control = data_dict['phase_control']
        else:
            training_phase_control = TrainingPhaseControlFlowRoughness(params)

        outlier_loss_detect = data_dict['outlier_loss_detect']
        if outlier_loss_detect is None or params["reset_outlier_detect"]:
            outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)

    else:
        epoch_start = 0
        history_train, history_test = collections.defaultdict(list), collections.defaultdict(list)
        best_performance = {"loss_on_test": {"image_loss": np.inf}}
        training_phase_control = TrainingPhaseControlFlowRoughness(params)
        outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)

    loss_tracker_train = TrackLossValue(batch_size=params["batch_size_train"], loss_values_dict_epoch=history_train)
    loss_tracker_test = TrackLossValue(batch_size=params["batch_size_test"], loss_values_dict_epoch=history_test)

    if params["reset_best_performance"]:
        best_performance = {"loss_on_test": {"image_loss": np.inf}}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

    print("flip_high_rough:", training_phase_control.flip_high_rough,
          "flip_low_rough:", training_phase_control.flip_low_rough,
          "relative_penalty_for_roughness:", training_phase_control.relative_penalty_for_flow,
          "target_rough:", training_phase_control.target_rough,
          "flip_remaining:", training_phase_control.flip_remaining)

    model_failed = False

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)

        accumulative_step = 0

        overlap_voxel = 0
        total_voxel = 0

        overlap_voxel_original = 0
        total_voxel_original = 0

        relative_penalty_for_flow = training_phase_control.relative_penalty_for_flow
        weight_image_based_loss, weight_flow_based_loss = 100 / relative_penalty_for_flow, relative_penalty_for_flow

        if not params["phase_shift"]:
            weight_image_based_loss, weight_flow_based_loss = 1, 1
        print("weight_image_based_loss, weight_flow_based_loss:", weight_image_based_loss, weight_flow_based_loss)
        training_phase_control.show_status()

        model.train()
        for i, batch_sample in enumerate(train_loader):

            if model_failed:
                continue

            if random.uniform(0, 1) < params["ratio_same_to_same"]:
                batch_sample = set_fixed_and_registered_as_the_same(batch_sample)

            if random.uniform(0, 1) < params["ratio_apply_translate"]:
                batch_sample = apply_translate_augmentation(batch_sample, params)

            fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = batch_sample
            # tensors in shape [B, 1, L, L, L]

            for channel, whether_important in enumerate(importance_list):
                if whether_important and params["use_penalty_weight"]:
                    penalty_weight_tensor[channel] = penalty_weight_tensor[channel] * params["weight_for_important"]

            if not params["use_penalty_weight"]:
                penalty_weight_tensor = None

            registration_flow_refined, registered_image_tensor = model(
                moving_image=moving_image_tensor, fixed_image=fixed_image_tensor, registration_flow_raw=None)
            # (B, 3, L, L, L), (B, 3, L, L, L)

            vessel_array_moving = moving_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_fix = fixed_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_registered = registered_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_fix = np.array(vessel_array_fix > 0, 'float32')
            vessel_array_registered = np.array(vessel_array_registered > 0, 'float32')
            vessel_array_moving = np.array(vessel_array_moving > 0, 'float32')
            overlap_voxel += np.sum(vessel_array_fix * vessel_array_registered)
            total_voxel += np.sum(vessel_array_registered + vessel_array_fix)
            overlap_voxel_original += np.sum(vessel_array_moving * vessel_array_fix)
            total_voxel_original += np.sum(vessel_array_moving + vessel_array_fix)

            image_loss, flow_loss, loss_value_dict = combination_loss(
                registered_image_tensor, fixed_image_tensor, penalty_weight_tensor, registration_flow_refined, params)

            loss_tracker_train.update_loss_values_batch(loss_value_dict)

            loss = weight_flow_based_loss * flow_loss + weight_image_based_loss * image_loss

            float_loss = loss.detach().float().cpu().data
            if params["use_outlier_loss_detect"]:
                loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss
            else:
                loss_status = True
            if i % params["visualize_interval"] == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), float_loss))

            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                model_failed = True  # detect failure inside epoch
                continue

            if not loss_status:  # an outlier is detected
                std_in_queue, ave_in_queue = outlier_loss_detect.get_std_and_ave_in_queue()
                loss = loss / abs(float_loss - ave_in_queue) * std_in_queue / 10  # reduce the weight for the loss

            accumulative_step += 1

            loss = loss / params["accumulate_step"]
            loss.backward()
            if (accumulative_step + 1) % params["accumulate_step"] == 0:
                optimizer.step()
                optimizer.zero_grad()

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
            model_failed = False  # rolled back
            continue

        loss_tracker_train.update_loss_values_epoch()

        dice_train = overlap_voxel * 2 / total_voxel
        dice_train_original = overlap_voxel_original * 2 / total_voxel_original
        print("\nDice Train Original:", dice_train_original)
        print("Dice Train Registered:", dice_train, '\n')
        current_performance = {"loss_on_test": loss_tracker_test.return_current_performance(),
                               "loss_on_train": loss_tracker_train.return_current_performance()}
        print("\ncurrent_performance train:")
        print(current_performance["loss_on_train"], '\n')

        print("\tEvaluating")

        evaluate(model, test_loader, params, training_phase_control, loss_tracker_test)

        history_train = loss_tracker_train.return_history()
        history_test = loss_tracker_test.return_history()

        history = [history_train, history_test]

        if current_performance["loss_on_test"]["image_loss"] < best_performance["loss_on_test"]["image_loss"]:
            print("\nNew best model_guided at loss_ave_test:", current_performance["loss_on_test"]["image_loss"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

        if training_phase_control.changed_phase_in_last_epoch:
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                            training_phase_control=training_phase_control,
                            special_name="flip_remaining:" + str(
                                training_phase_control.flip_remaining) + '_' + training_phase_control.previous_phase,
                            outlier_loss_detect=outlier_loss_detect)
            print("updating backup model at performance:", current_performance["loss_on_test"]["image_loss"])
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                            best=False,
                            training_phase_control=training_phase_control,
                            special_name="backup",
                            outlier_loss_detect=outlier_loss_detect)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, params, training_phase_control, loss_tracker_test):

    roughness = 0

    overlap_voxel = 0
    total_voxel = 0

    overlap_voxel_original = 0
    total_voxel_original = 0

    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(test_loader):

            if random.uniform(0, 1) < params["ratio_same_to_same"]:
                batch_sample = set_fixed_and_registered_as_the_same(batch_sample)

            if random.uniform(0, 1) < params["ratio_apply_translate"]:
                batch_sample = apply_translate_augmentation(batch_sample, params)

            fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = batch_sample
            # tensors in shape [B, 3, L, L, L], [B, 3, L, L, L], [B, 1, L, L, L]

            for channel, whether_important in enumerate(importance_list):
                if whether_important:
                    penalty_weight_tensor[channel] = penalty_weight_tensor[channel] * params["weight_for_important"]
            if not params["use_penalty_weight"]:
                penalty_weight_tensor = None

            registration_flow_refined, registered_image_tensor = model(
                moving_image=moving_image_tensor, fixed_image=fixed_image_tensor, registration_flow_raw=None)
            # (B, 3, L, L, L), (B, 3, L, L, L)

            vessel_array_moving = moving_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_fix = fixed_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_registered = registered_image_tensor[:, 1].cpu().detach().numpy()
            vessel_array_fix = np.array(vessel_array_fix > 0, 'float32')
            vessel_array_registered = np.array(vessel_array_registered > 0, 'float32')
            vessel_array_moving = np.array(vessel_array_moving > 0, 'float32')

            overlap_batch = np.sum(vessel_array_fix * vessel_array_registered)
            total_batch = np.sum(vessel_array_registered + vessel_array_fix)
            overlap_original_batch = np.sum(vessel_array_moving * vessel_array_fix)
            total_original_batch = np.sum(vessel_array_moving + vessel_array_fix)

            overlap_voxel += overlap_batch
            total_voxel += total_batch
            overlap_voxel_original += overlap_original_batch
            total_voxel_original += total_original_batch

            jacobi_determinant_tensor = flow_based_loss.get_jacobi_high_precision(
                registration_flow_refined, precision=params["precision_for_jacobi"])

            image_loss, flow_loss, loss_value_dict = combination_loss(
                registered_image_tensor, fixed_image_tensor, penalty_weight_tensor, registration_flow_refined, params,
                jacobi_determinant_tensor=jacobi_determinant_tensor)

            loss_tracker_test.update_loss_values_batch(loss_value_dict)

            roughness += torch.max(jacobi_determinant_tensor).detach().float().cpu().data
            if torch.min(jacobi_determinant_tensor) < 0:  # give very high penalty on negative
                roughness += 20

            if i % params["visualize_interval"] == 0:
                print("\tStep [%d/%d]   dice_original_batch=%.4f,   dice_registered_batch=%.4f" %
                      (i + 1, len(test_loader), overlap_original_batch * 2 / total_original_batch,
                       overlap_batch * 2 / total_batch))

        loss_tracker_test.update_loss_values_epoch()
        roughness = roughness / len(test_loader)
        if training_phase_control is not None:
            training_phase_control.get_new_relative_penalty_for_flow(roughness)

        dice_test = overlap_voxel * 2 / total_voxel
        dice_test_original = overlap_voxel_original * 2 / total_voxel_original

        print("\n\nDice Test Original:", dice_test_original, "    Dice Test Registered:", dice_test,
              "    Current Roughness Test:", roughness, '\n')
        print("\ncurrent_performance test:")
        print(loss_tracker_test.return_current_performance())

        return dice_test, roughness
    
    
def combination_loss(registered_image_tensor, fixed_image_tensor, penalty_weight_tensor, 
                     registration_flow_refined, params, jacobi_determinant_tensor=None):
    """
    
    :param jacobi_determinant_tensor:
    :param registration_flow_refined: [N, 3, L, L, L]
    :param registered_image_tensor: [N, 3, L, L, L]
    :param fixed_image_tensor: [N, 3, L, L, L]
    :param penalty_weight_tensor: [N, 1, L, L, L], or None
    :param params: 
    :return: image_loss, flow_loss, loss_value_dict
    """
    channel_weight = params["channel_weight"]
    
    loss_value_dict = {}
    
    if params["use_ncc_loss"]:
        ncc_loss_0 = image_based_loss.weighted_ncc_loss(
            registered_image_tensor[:, 0: 1], fixed_image_tensor[:, 0: 1],
            penalty_weight_tensor, stride_step=params["ncc_stride"])
        ncc_loss_1 = image_based_loss.weighted_ncc_loss(
            registered_image_tensor[:, 1: 2], fixed_image_tensor[:, 1: 2],
            penalty_weight_tensor, stride_step=params["ncc_stride"])
        ncc_loss_2 = image_based_loss.weighted_ncc_loss(
            registered_image_tensor[:, 2: 3], fixed_image_tensor[:, 2: 3],
            penalty_weight_tensor, stride_step=params["ncc_stride"])
        ncc_loss_3 = image_based_loss.weighted_ncc_loss(
            registered_image_tensor[:, 3: 4], fixed_image_tensor[:, 3: 4],
            penalty_weight_tensor, stride_step=params["ncc_stride"])
        ncc_loss = ncc_loss_0 * channel_weight[0] + ncc_loss_1 * channel_weight[1] + ncc_loss_2 * channel_weight[2] \
            + ncc_loss_3 * channel_weight[3]
        
        loss_value_dict["ncc_loss_0"] = ncc_loss_0.detach().float().cpu().data
        loss_value_dict["ncc_loss_1"] = ncc_loss_1.detach().float().cpu().data
        loss_value_dict["ncc_loss_2"] = ncc_loss_2.detach().float().cpu().data
        loss_value_dict["ncc_loss_3"] = ncc_loss_2.detach().float().cpu().data
        loss_value_dict["ncc_loss"] = ncc_loss.detach().float().cpu().data
        
    else:
        ncc_loss = 0
        
    if params["use_mae_normalized_loss"]:
        mae_loss_0 = image_based_loss.mae_loss_normalized(registered_image_tensor[:, 0], fixed_image_tensor[:, 0])
        mae_loss_1 = image_based_loss.mae_loss_normalized(registered_image_tensor[:, 1], fixed_image_tensor[:, 1])
        mae_loss_2 = image_based_loss.mae_loss_normalized(registered_image_tensor[:, 2], fixed_image_tensor[:, 2])
        mae_loss_3 = image_based_loss.mae_loss_normalized(registered_image_tensor[:, 3], fixed_image_tensor[:, 3])
        mae_loss = mae_loss_0 * channel_weight[0] + mae_loss_1 * channel_weight[1] + mae_loss_2 * channel_weight[2] \
            + mae_loss_3 * channel_weight[3] 
        loss_value_dict["mae_loss_0"] = mae_loss_0.detach().float().cpu().data
        loss_value_dict["mae_loss_1"] = mae_loss_1.detach().float().cpu().data
        loss_value_dict["mae_loss_2"] = mae_loss_2.detach().float().cpu().data
        loss_value_dict["mae_loss_3"] = mae_loss_3.detach().float().cpu().data
        loss_value_dict["mae_loss"] = mae_loss.detach().float().cpu().data
    else:
        mae_loss = 0
        
    if params["use_mse_normalized_loss"]:
        mse_loss_0 = image_based_loss.mse_loss_normalized(registered_image_tensor[:, 0], fixed_image_tensor[:, 0])
        mse_loss_1 = image_based_loss.mse_loss_normalized(registered_image_tensor[:, 1], fixed_image_tensor[:, 1])
        mse_loss_2 = image_based_loss.mse_loss_normalized(registered_image_tensor[:, 2], fixed_image_tensor[:, 2])
        mse_loss_3 = image_based_loss.mse_loss_normalized(registered_image_tensor[:, 3], fixed_image_tensor[:, 3])
        mse_loss = mse_loss_0 * channel_weight[0] + mse_loss_1 * channel_weight[1] + mse_loss_2 * channel_weight[2] \
            + mse_loss_3 * channel_weight[3]
        loss_value_dict["mse_loss_0"] = mse_loss_0.detach().float().cpu().data
        loss_value_dict["mse_loss_1"] = mse_loss_1.detach().float().cpu().data
        loss_value_dict["mse_loss_2"] = mse_loss_2.detach().float().cpu().data
        loss_value_dict["mse_loss_3"] = mse_loss_3.detach().float().cpu().data
        loss_value_dict["mse_loss"] = mse_loss.detach().float().cpu().data
    else:
        mse_loss = 0

    if params["use_flow_based_loss"]:
        gradient_loss = flow_based_loss.gradient_loss_l2(registration_flow_refined)
        loss_value_dict["gradient_loss"] = gradient_loss.detach().float().cpu().data

        if jacobi_determinant_tensor is None:
            jacobi_determinant_tensor = flow_based_loss.get_jacobi_high_precision(
                registration_flow_refined, precision=params["precision_for_jacobi"])
        if params["include_negative_jacobi_loss"]:
            negative_jacobi_loss = flow_based_loss.negative_jacobi_loss(jacobi_determinant_tensor)
            loss_value_dict["negative_jacobi_loss"] = negative_jacobi_loss.detach().float().cpu().data
        else:
            negative_jacobi_loss = 0
        flow_tension_loss = flow_based_loss.flow_tension_loss(jacobi_determinant_tensor)
        flow_tension_loss = flow_tension_loss * params["tension_loss_augment_ratio"]
        flow_loss = negative_jacobi_loss + flow_tension_loss + gradient_loss
        loss_value_dict["flow_tension_loss"] = flow_tension_loss.detach().float().cpu().data
        loss_value_dict["flow_loss"] = flow_loss.detach().float().cpu().data
    else:
        flow_loss = 0
    
    ratio_ncc_mae_mse = params["ratio_ncc_mae_mse"]
    
    image_loss = ratio_ncc_mae_mse[0] * ncc_loss + ratio_ncc_mae_mse[1] * mae_loss + ratio_ncc_mae_mse[2] * mse_loss
    loss_value_dict["image_loss"] = image_loss.detach().float().cpu().data

    return image_loss, flow_loss, loss_value_dict


class TrackLossValue:
    """
    each time, receive loss_value_dict, then normalize with batch_size and store
    """
    def __init__(self, batch_size=1, loss_values_dict_epoch=None):
        self.loss_values_dict_batch = None
        self.loss_values_dict_epoch = loss_values_dict_epoch
        self.batch_size = batch_size

    def update_loss_values_batch(self, loss_value_dict):  # call it for each batch
        if self.loss_values_dict_batch is None:
            self.loss_values_dict_batch = collections.defaultdict(list)
        for key, value in loss_value_dict.items():
            if value is None:
                value = 0
            self.loss_values_dict_batch[key].append(value / self.batch_size)

    def update_loss_values_epoch(self):  # call it when epoch end
        if self.loss_values_dict_epoch is None:
            self.loss_values_dict_epoch = collections.defaultdict(list)
        for key, value in self.loss_values_dict_batch.items():
            self.loss_values_dict_epoch[key].append(np.mean(value))
        self._clear_batch_dict()

    def return_history(self):
        return self.loss_values_dict_epoch

    def return_current_performance(self):
        current_performance_dict = {}
        for key, value in self.loss_values_dict_epoch.items():
            current_performance_dict[key] = value[-1]
        return current_performance_dict

    def _clear_batch_dict(self):
        self.loss_values_dict_batch = None
