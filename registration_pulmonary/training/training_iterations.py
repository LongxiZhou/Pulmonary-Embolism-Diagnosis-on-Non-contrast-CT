"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import registration_pulmonary.models.model_no_landmark as main_model
import registration_pulmonary.training.dataset_and_dataloader as dataset_and_loader
import registration_pulmonary.loss_functions.image_based_loss as image_based_loss
import registration_pulmonary.loss_functions.flow_based_loss as flow_based_loss
from registration_pulmonary.utlis.loss_and_phase_control import OutlierLossDetect, TrainingPhaseControlFlowRoughness
from registration_pulmonary.simulation_and_augmentation.process_data_augmentation import apply_translate_augmentation


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    image_size = (params["image_length"], params["image_length"], params["image_length"])

    model = main_model.RefineRegistrationFlow(
        image_size=image_size, num_landmark=params["num_landmark"],
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
        augment=params["augment"], drop_last=True)

    test_loader = dataset_and_loader.DataLoaderRegistration(
        original_sample_dataset_test, params["batch_size_test"], shuffle=False, mode='test',
        augment=params["augment"], drop_last=False)

    print("there are:", len(original_sample_dataset_train), "training ct pairs")
    print("there are:", len(original_sample_dataset_test), "testing ct pairs")

    train_loop(model, optimizer, train_loader, test_loader, params)


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
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
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
        history = data_dict["history"]
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
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf}
        training_phase_control = TrainingPhaseControlFlowRoughness(params)
        outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)

    if params["reset_best_performance"]:
        best_performance = {"loss_ave_test": np.inf}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

    print("flip_high_rough:", training_phase_control.flip_high_rough,
          "flip_low_rough:", training_phase_control.flip_low_rough,
          "relative_penalty_for_roughness:", training_phase_control.relative_penalty_for_flow,
          "target_rough:", training_phase_control.target_rough,
          "flip_remaining:", training_phase_control.flip_remaining)

    model_failed = False

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)

        loss_ave = 0
        loss_ave_ncc = 0
        loss_ave_negative_jacobi = 0
        loss_ave_flow_tension = 0

        accumulative_step = 0

        relative_penalty_for_flow = training_phase_control.relative_penalty_for_flow
        weight_image_based_loss, weight_flow_based_loss = 100 / relative_penalty_for_flow, relative_penalty_for_flow

        if not params["use_flow_based_loss"]:
            weight_image_based_loss, weight_flow_based_loss = 1, 0

        print("weight_image_based_loss, weight_flow_based_loss:", weight_image_based_loss, weight_flow_based_loss)
        training_phase_control.show_status()

        model.train()
        for i, batch_sample in enumerate(train_loader):

            if model_failed:
                continue

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
            # (B, 3, L, L, L), (B, 1, L, L, L)

            if params["ratio_easy"] < 1:
                ncc_loss = image_based_loss.weighted_ncc_loss(
                    registered_image_tensor, fixed_image_tensor,
                    penalty_weight_tensor, stride_step=params["ncc_stride"])
            else:
                ncc_loss = image_based_loss.mae_loss(registered_image_tensor, fixed_image_tensor)

            jacobi_determinant_tensor = flow_based_loss.get_jacobi_high_precision(
                registration_flow_refined, precision=params["precision_for_jacobi"])

            if not params["use_flow_based_loss"]:
                flow_loss = 0
                float_negative_jacobi_loss = 0
                float_flow_tension_loss = 0
            else:
                negative_jacobi_loss = flow_based_loss.negative_jacobi_loss(jacobi_determinant_tensor)
                flow_tension_loss = flow_based_loss.flow_tension_loss(jacobi_determinant_tensor)
                flow_loss = negative_jacobi_loss + flow_tension_loss

                float_negative_jacobi_loss = negative_jacobi_loss.detach().float().cpu().data
                float_flow_tension_loss = flow_tension_loss.detach().float().cpu().data

            loss = weight_flow_based_loss * flow_loss + weight_image_based_loss * ncc_loss

            float_loss = loss.detach().float().cpu().data
            float_ncc_loss = ncc_loss.detach().float().cpu().data

            if params["use_outlier_loss_detect"]:
                loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss
            else:
                loss_status = True

            if i % 10 == 0:
                with torch.no_grad():
                    if params["ratio_easy"] < 1:
                        ncc_loss_original = image_based_loss.weighted_ncc_loss(
                            moving_image_tensor, fixed_image_tensor, penalty_weight_tensor,
                            stride_step=params["ncc_stride"], win_length=params["ncc_window_length"])
                        ncc_loss_optimal = image_based_loss.weighted_ncc_loss(
                            fixed_image_tensor, fixed_image_tensor, penalty_weight_tensor,
                            stride_step=params["ncc_stride"], win_length=params["ncc_window_length"])
                    else:
                        ncc_loss_original = image_based_loss.mae_loss(
                            moving_image_tensor, fixed_image_tensor)
                        ncc_loss_optimal = image_based_loss.mae_loss(
                            fixed_image_tensor, fixed_image_tensor)

                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss),
                      'ncc loss:', float_ncc_loss.detach().float().cpu().data,
                      "; ncc_loss_original:", ncc_loss_original.detach().float().cpu().data,
                      "; ncc_loss_optimal:", ncc_loss_optimal.detach().float().cpu().data,
                      '; flow tension loss:', float_flow_tension_loss,
                      '; negative jacobi loss:', float_negative_jacobi_loss)
                del ncc_loss_original, ncc_loss_optimal

            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                model_failed = True  # detect failure inside epoch
                continue

            if not loss_status:  # an outlier is detected
                std_in_queue, ave_in_queue = outlier_loss_detect.get_std_and_ave_in_queue()
                loss = loss / abs(float_loss - ave_in_queue) * std_in_queue / 10  # reduce the weight for the loss

            accumulative_step += 1

            loss_ave += float_loss
            loss_ave_flow_tension += float_flow_tension_loss
            loss_ave_negative_jacobi += float_negative_jacobi_loss
            loss_ave_ncc += float_ncc_loss

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

        loss_ave = loss_ave / accumulative_step / params["batch_size_train"]
        loss_ave_ncc = loss_ave_ncc / accumulative_step / params["batch_size_train"]
        loss_ave_negative_jacobi = loss_ave_negative_jacobi / accumulative_step / params["batch_size_train"]
        loss_ave_flow_tension = loss_ave_flow_tension / accumulative_step / params["batch_size_train"]

        print("\nloss average on each CT scan training:", loss_ave)
        print("\nncc loss average on each CT scan training:", loss_ave_ncc)
        print("\nnegative jacobi loss average on each CT scan training:", loss_ave_negative_jacobi)
        print("\nflow tension loss average on each CT scan training:", loss_ave_flow_tension)

        history["loss_average_on_each_scan_training"].append(loss_ave)
        history["ncc_loss_average_on_each_scan_training"].append(loss_ave_ncc)
        history["negative_jacobi_loss_average_on_each_scan_training"].append(loss_ave_negative_jacobi)
        history["flow_tension_loss_average_on_each_scan_training"].append(loss_ave_flow_tension)

        print("\tEvaluating")

        loss_ave_test, loss_ave_ncc_test, loss_ave_negative_jacobi_test, loss_ave_flow_tension_test = \
            evaluate(model, test_loader, params, training_phase_control, history)

        current_performance = {"loss_ave_train": loss_ave, "loss_ave_test": loss_ave_test,
                               "loss_ave_ncc_train": loss_ave_ncc,
                               "loss_ave_negative_jacobi_train": loss_ave_negative_jacobi,
                               "loss_ave_flow_tension_train": loss_ave_flow_tension,
                               "loss_ave_ncc_test": loss_ave_ncc_test,
                               "loss_ave_negative_jacobi_test": loss_ave_negative_jacobi_test,
                               "loss_ave_flow_tension_test": loss_ave_flow_tension_test,
                               "relative_penalty_for_flow": relative_penalty_for_flow}

        if current_performance["loss_ave_test"] < best_performance["loss_ave_test"]:
            print("\nNew best model_guided at loss_ave_test:", current_performance["loss_ave_test"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

        if training_phase_control.changed_phase_in_last_epoch:
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                            training_phase_control=training_phase_control,
                            special_name="flip_remaining:" + str(
                                training_phase_control.flip_remaining) + '_' + training_phase_control.previous_phase,
                            outlier_loss_detect=outlier_loss_detect)
            print("updating backup model at performance:", current_performance)
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                            best=False,
                            training_phase_control=training_phase_control,
                            special_name="backup",
                            outlier_loss_detect=outlier_loss_detect)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, params, training_phase_control, history):
    loss_ave_ncc_test = 0
    loss_ave_negative_jacobi_test = 0
    loss_ave_flow_tension_test = 0

    roughness = 0

    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(test_loader):

            batch_sample = apply_translate_augmentation(batch_sample, params)

            fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = batch_sample
            # tensors in shape [B, 1, L, L, L]

            for channel, whether_important in enumerate(importance_list):
                if whether_important:
                    penalty_weight_tensor[channel] = penalty_weight_tensor[channel] * params["weight_for_important"]

            registration_flow_refined, registered_image_tensor = model(
                moving_image=moving_image_tensor, fixed_image=fixed_image_tensor, registration_flow_raw=None)
            # (B, 3, L, L, L), (B, 1, L, L, L)

            if params["ratio_easy"] < 1:
                ncc_loss = image_based_loss.weighted_ncc_loss(
                    registered_image_tensor, fixed_image_tensor,
                    penalty_weight_tensor, stride_step=params["ncc_stride"])
            else:
                ncc_loss = image_based_loss.mae_loss(registered_image_tensor, fixed_image_tensor)

            jacobi_determinant_tensor = flow_based_loss.get_jacobi_high_precision(
                registration_flow_refined, precision=params["precision_for_jacobi"])

            if not params["use_flow_based_loss"]:
                negative_jacobi_loss = 0
                flow_tension_loss = 0
            else:
                negative_jacobi_loss = flow_based_loss.negative_jacobi_loss(
                    jacobi_determinant_tensor).detach().float().cpu().data
                flow_tension_loss = flow_based_loss.flow_tension_loss(
                    jacobi_determinant_tensor).detach().float().cpu().data

            roughness += torch.max(jacobi_determinant_tensor).detach().float().cpu().data
            if torch.min(jacobi_determinant_tensor) < 0:  # give very high penalty on negative
                roughness += 20

            loss_ave_ncc_test += ncc_loss.detach().float().cpu().data
            loss_ave_negative_jacobi_test += negative_jacobi_loss
            loss_ave_flow_tension_test += flow_tension_loss

        roughness = roughness / len(test_loader)

        loss_ave_ncc_test = loss_ave_ncc_test / len(test_loader) / params["batch_size_test"]
        loss_ave_negative_jacobi_test = loss_ave_negative_jacobi_test / len(test_loader) / params["batch_size_test"]
        loss_ave_flow_tension_test = loss_ave_flow_tension_test / len(test_loader) / params["batch_size_test"]

        print("\nncc loss average on each CT scan testing:", loss_ave_ncc_test)
        print("\nnegative jacobi loss average on each CT scan testing:", loss_ave_negative_jacobi_test)
        print("\nflow tension loss average on each CT scan testing:", loss_ave_flow_tension_test)

        relative_penalty_for_flow = training_phase_control.relative_penalty_for_flow
        weight_image_based_loss, weight_flow_based_loss = 100 / relative_penalty_for_flow, relative_penalty_for_flow

        loss_ave_test = loss_ave_ncc_test * weight_image_based_loss + (
                loss_ave_negative_jacobi_test + loss_ave_flow_tension_test) * weight_flow_based_loss

        history["loss_average_on_each_scan_testing"].append(loss_ave_test)
        history["ncc_loss_average_on_each_scan_testing"].append(loss_ave_ncc_test)
        history["negative_jacobi_loss_average_on_each_scan_testing"].append(loss_ave_negative_jacobi_test)
        history["flow_tension_loss_average_on_each_scan_testing"].append(loss_ave_flow_tension_test)

        training_phase_control.get_new_relative_penalty_for_flow(roughness)

        print("\n\n Current Roughness:", roughness, '\n\n')

        return loss_ave_test, loss_ave_ncc_test, loss_ave_negative_jacobi_test, loss_ave_flow_tension_test
