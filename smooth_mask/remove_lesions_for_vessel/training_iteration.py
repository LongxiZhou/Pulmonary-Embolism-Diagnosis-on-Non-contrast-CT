"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import models.Unet_3D.U_net_Model_3D as U_net_Models
import smooth_mask.remove_lesions_for_vessel.loss_function as loss_function
import smooth_mask.remove_lesions_for_vessel.dataset_and_dataloader as dataset_and_loader


def training(params):
    # -------------------------------------------------------
    # form model
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    if params["model_size"] == 'large':
        model = U_net_Models.UNet3D(params["in_channels"], params["out_channels"], params["init_features"])
    elif params["model_size"] == 'median':
        model = U_net_Models.UNet3DSimple(params["in_channels"], params["out_channels"], params["init_features"])
    else:
        assert params["model_size"] == 'small'
        model = U_net_Models.UNet3DSimplest(params["in_channels"], params["out_channels"], params["init_features"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # -------------------------------------------------------
    # form dataset
    lesion_dataset = dataset_and_loader.SimulateLesionDataset(
        difficulty_level=params["difficulty_level"], penalty_range=params["penalty_range"],
        lesion_top_dict=params["lesion_top_dict"])

    train_dataset = dataset_and_loader.SmoothMaskDataset(
        params["list_train_data_dir"], mode='train', test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"], sample_interval=params["sample_interval"])

    test_dataset = dataset_and_loader.SmoothMaskDataset(
        params["list_test_data_dir"], mode='test', test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"], sample_interval=params["sample_interval"])

    print("train data_dirs:", params["list_train_data_dir"], len(train_dataset))
    print("there are:", len(train_dataset), "training samples")
    print("there are:", len(test_dataset), "testing samples")
    print("difficulty will be:", params["difficulty_level"])

    # -------------------------------------------------------
    # form dataloader
    train_loader = dataset_and_loader.SmoothMaskDataloader(
        train_dataset, lesion_dataset, params["batch_size"], True, num_workers=params["num_workers"],
        drop_last=params["drop_last"], show=True, mode='train', num_test_replicate=None,
        num_lesion_applied=params["num_lesion_applied"], random_augment=params["random_augment"])

    test_loader = dataset_and_loader.SmoothMaskDataloader(
        test_dataset, lesion_dataset, params["batch_size"], True, num_workers=params["num_workers"],
        drop_last=False, show=True, mode='test', num_test_replicate=params["num_test_replicate"],
        num_lesion_applied=params["num_lesion_applied"], random_augment=params["random_augment"])

    train_loop(model, optimizer, train_loader, test_loader, params)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
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
    print("saving model_smooth to path:", save_path)
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_performance': best_performance,
        'current_performance': current_performance,
        'phase_control': training_phase_control,
        'outlier_loss_detect': outlier_loss_detect,
        'difficulty_level': params['difficulty_level'],
    }, save_path, _use_new_zipfile_serialization=False)


def train_loop(model, optimizer, train_loader, test_loader, params=None):
    from med_transformer.utlis import OutlierLossDetect, TrainingPhaseControl
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
            training_phase_control = TrainingPhaseControl(params)

        outlier_loss_detect = data_dict['outlier_loss_detect']
        if outlier_loss_detect is None or params["reset_outlier_detect"]:
            outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)
    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}
        training_phase_control = TrainingPhaseControl(params)
        outlier_loss_detect = OutlierLossDetect(30, 3, 3, 10)

    if params["reset_best_performance"]:
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))
    print("difficulty_level:", params["difficulty_level"])

    softmax_layer = torch.nn.Softmax(dim=1)

    print("flip_recall:", training_phase_control.flip_recall, "flip_precision:", training_phase_control.flip_precision,
          "base_relative:", training_phase_control.base_relative,
          "base_recall", training_phase_control.base_recall,
          "base_precision", training_phase_control.base_precision,
          "max_performance_recall:", training_phase_control.max_performance_recall,
          "max_performance_precision:", training_phase_control.max_performance_precision)
    if params["normalize_by_positive_voxel"]:
        print("loss normalize protocol:", params["normalize_protocol"], "normalize base:", params["normalize_base"])

    model_failed = False

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)

        loss_ave = 0
        total_lesion_voxel = 0
        num_true_positive = 0
        num_false_positive = 0

        accumulative_step = 0

        relative_false_positive_penalty = training_phase_control.relative_false_positive_penalty
        # higher means model give less false positives, at the expense of more false negative
        # be careful about the meaning for each channel:
        # "relative_false_positive_penalty" should be the channel for negative prediction
        class_balance = [100 / relative_false_positive_penalty, relative_false_positive_penalty]
        print("class balance:", class_balance)
        training_phase_control.show_status()

        model.train()
        for i, batch_sample in enumerate(train_loader):
            # batch_sample:     model_input_tensor, model_gt_output_tensor, weight_tensor   on CPU
            #                   [B, 1, x, y, z]       [B, 2, x, y, z]      [B, 2, x, y, z]

            if model_failed:
                continue

            model_input_tensor, model_gt_output_tensor, weight_tensor = batch_sample

            model_input_tensor = model_input_tensor.cuda()
            model_gt_output_tensor = model_gt_output_tensor.cuda()
            weight_tensor = weight_tensor.cuda()

            segmentation_before_softmax = model(model_input_tensor)
            # [B, 2, x, y, z]

            loss = loss_function.cross_entropy_get_blood_region(
                segmentation_before_softmax, model_gt_output_tensor, weight_tensor, class_balance)

            if params["normalize_by_positive_voxel"]:
                if params["normalize_protocol"] == 'log':
                    loss = loss / torch.log(
                        torch.sum(model_gt_output_tensor.detach()[:, 0, :, :]) + params["normalize_base"])
                elif params["normalize_protocol"] == 'sqrt':
                    loss = loss / torch.sqrt(
                        torch.sum(model_gt_output_tensor.detach()[:, 0, :, :]) + params["normalize_base"])
                elif params["normalize_protocol"] == 'abs':
                    loss = loss / torch.abs(
                        torch.sum(model_gt_output_tensor.detach()[:, 0, :, :]) + params["normalize_base"])
                else:
                    raise ValueError

            if i % params['visualize_interval'] == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss))

            loss_sample = loss.detach().float().cpu().data / params["batch_size"]

            loss_status = outlier_loss_detect.update_new_loss(loss_sample)  # True for good loss, False for bad loss
            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                del loss, segmentation_before_softmax, weight_tensor, model_gt_output_tensor, model_input_tensor
                model_failed = True  # detect failure inside epoch
                continue

            if not loss_status:  # an outlier is detected
                std_in_queue, ave_in_queue = outlier_loss_detect.get_std_and_ave_in_queue()
                loss = loss / abs(loss_sample - ave_in_queue) * std_in_queue
                # reduce the weight for the loss, like 10 std outlier, weight for this batch reduce to 1/10

            accumulative_step += 1

            loss_ave += loss_sample * params["batch_size"]
            loss = loss / params["accumulate_step"]
            loss.backward()
            if (accumulative_step + 1) % params["accumulate_step"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            segmentation_before_softmax = segmentation_before_softmax.detach()

            segment_probability_lesion = softmax_layer(segmentation_before_softmax).cpu().numpy()[:, 0, :, :]
            segment_mask_lesion = np.array(segment_probability_lesion > 0.5, 'float32')
            lesion_mask_gt = model_gt_output_tensor.detach().cpu().numpy()[:, 0, :, :]

            if not np.min(lesion_mask_gt) == 0 and np.max(lesion_mask_gt) == 1:
                print("range for lesion_mask_gt:", np.min(lesion_mask_gt), np.max(lesion_mask_gt))

            overlap_count_batch = np.sum(lesion_mask_gt * segment_mask_lesion)

            num_true_positive += overlap_count_batch
            total_lesion_voxel += np.sum(lesion_mask_gt)
            num_false_positive += np.sum(segment_mask_lesion) - overlap_count_batch

            if i == 0 and epoch == 0:
                print("size for model_input_tensor", model_input_tensor.size())
                print("size for model_gt_output_tensor", model_gt_output_tensor.size())
                print("initial class balance:", class_balance)

            del segmentation_before_softmax, weight_tensor, model_gt_output_tensor, model_input_tensor

        if model_failed:
            print("failure model, roll back to previous best version")
            backup_model_path = os.path.join(params["checkpoint_dir"], "best_" + params["saved_model_filename"])

            data_dict = torch.load(backup_model_path)
            if type(model) == nn.DataParallel:
                model.module.load_state_dict(data_dict["state_dict"])
            else:
                model.load_state_dict(data_dict["state_dict"])

            training_phase_control = data_dict['phase_control']
            outlier_loss_detect = data_dict['outlier_loss_detect']

            model_failed = False  # rolled back
            continue

        recall = num_true_positive / total_lesion_voxel
        precision = num_true_positive / (num_true_positive + num_false_positive)

        if recall <= 0 or precision <= 0:
            dice = 0
        else:
            dice = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / train_loader.num_samples
        print("\nloss average on each CT scan training:", loss_ave)
        print("recall on training:", recall)
        print("precision on training:", precision)
        print("dice on training:", dice, '\n')
        history["loss_average_on_each_scan_training"].append(loss_ave)
        history["recall_for_each_training_epoch"].append(recall)
        history["precision_for_each_training_epoch"].append(precision)

        print("\tEvaluating")
        loss_ave_test, recall_test, precision_test, dice_test = \
            evaluate(model, test_loader, training_phase_control, history)

        current_performance = {"loss_ave_train": loss_ave, "loss_ave_test": loss_ave_test,
                               "recall_train": recall, "recall_test": recall_test,
                               "precision_train": precision, "precision_test": precision_test,
                               "dice_train": dice, "dice_test": dice_test,
                               "relative_false_positive_penalty": relative_false_positive_penalty}

        if training_phase_control.changed_phase_in_last_epoch:
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                            training_phase_control=training_phase_control,
                            special_name="level_" + str(params["difficulty_level"]) + '_dice_' +
                                         str(dice_test)[0: 5] + '_' + training_phase_control.previous_phase,
                            outlier_loss_detect=outlier_loss_detect)

        if current_performance["dice_test"] > best_performance["dice_test"]:
            print("\nNew best model_guided at dice test:", current_performance["dice_test"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

        if params["save_interval"] is not None:
            if epoch % params["save_interval"] == 0:
                save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                                best=False, special_name='epoch_' + str(epoch) + '_precision-' + str(precision_test) +
                                                         '_recall-' + str(recall_test),
                                training_phase_control=training_phase_control, outlier_loss_detect=outlier_loss_detect)

    print("Training finished")
    print("best_performance:", best_performance)


def evaluate(model, test_loader, training_phase_control, history):
    loss_ave = 0
    total_lesion_voxel = 0
    num_true_positive = 0
    num_false_positive = 0
    relative_false_positive_penalty = training_phase_control.relative_false_positive_penalty
    class_balance = [relative_false_positive_penalty, 100 / relative_false_positive_penalty]

    softmax_layer = torch.nn.Softmax(dim=1)

    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(test_loader):
            model_input_tensor, model_gt_output_tensor, weight_tensor = batch_sample
            model_input_tensor = model_input_tensor.cuda()
            model_gt_output_tensor = model_gt_output_tensor.cuda()
            weight_tensor = weight_tensor.cuda()

            segmentation_before_softmax = model(model_input_tensor)
            # [B, 2, x, y, z]

            loss = loss_function.cross_entropy_get_blood_region(
                segmentation_before_softmax, model_gt_output_tensor, weight_tensor, class_balance)

            loss_ave += loss.detach().float().cpu().data

            segment_probability_lesion = softmax_layer(segmentation_before_softmax).cpu().numpy()[:, 0, :, :]
            segment_mask_lesion = np.array(segment_probability_lesion > 0.5, 'float32')
            lesion_mask_gt = model_gt_output_tensor.detach().cpu().numpy()[:, 0, :, :]

            overlap_count_batch = np.sum(lesion_mask_gt * segment_mask_lesion)

            num_true_positive += overlap_count_batch
            total_lesion_voxel += np.sum(lesion_mask_gt)
            num_false_positive += np.sum(segment_mask_lesion) - overlap_count_batch

        recall = num_true_positive / total_lesion_voxel
        precision = num_true_positive / (num_true_positive + num_false_positive + 0.001)

        if recall <= 0 or precision <= 0:
            dice_test = 0
        else:
            dice_test = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / test_loader.num_samples
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
