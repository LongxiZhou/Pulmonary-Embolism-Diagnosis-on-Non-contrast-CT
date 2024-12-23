"""
Call function "training" to start the training.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import collections
import pulmonary_embolism_v2.transformer_PE_4D.model_transformer as model_transformer
import pulmonary_embolism_v2.transformer_PE_4D.loss_functions as loss_function
import pulmonary_embolism_v2.transformer_PE_4D.dataset_and_loader as dataset_and_loader
import med_transformer.utlis as utlis


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

    train_dataset = dataset_and_loader.PEDatasetSimulateClot(
        params["train_data_dir"],
        params["list-clot_sample_dict_dir"],
        mode='train',
        test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"],
        important_file_name=params["importance_file_name"]
    )

    test_dataset = dataset_and_loader.PEDatasetSimulateClot(
        params["test_data_dir"],
        params["list-clot_sample_dict_dir"],
        mode='test',
        test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"],
        important_file_name=params["importance_file_name"]
    )

    print("train:", params["train_data_dir"], len(train_dataset))
    print("there are:", len(train_dataset), "training ct scans")
    print("there are:", len(test_dataset), "testing ct scans")
    print("difficulty will be:", params["difficulty"])

    train_loader = \
        dataset_and_loader.DataLoaderSimulatedClot(train_dataset, params["batch_ct"], True, params["num_workers"],
                                                   True, True, 'train', params["num_prepared_dataset_train"],
                                                   params["num_prepared_dataset_test"], params["reuse_count"],
                                                   params["min_clot_count"])

    test_loader = \
        dataset_and_loader.DataLoaderSimulatedClot(test_dataset, params["batch_ct"], True, params["num_workers"],
                                                   False, True, 'test', params["num_prepared_dataset_train"],
                                                   params["num_prepared_dataset_test"], params["reuse_count"],
                                                   params["min_clot_count"])

    train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                  params["value_increase"], params["voxel_variance"])
    test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                 params["value_increase"], params["voxel_variance"])

    train_loop(model, optimizer, train_loader, test_loader, params)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True,
                    training_phase_control=None, special_name=None, outlier_loss_detect=None, value_increase=None):
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
            outlier_loss_detect = utlis.OutlierLossDetect(30, 3, 3, 10)

        if params["value_increase"] is not None:
            value_increase = list(params["value_increase"])
        else:
            value_increase = data_dict['value_increase']

    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf, "recall_test": 0, "precision_test": 0, "dice_test": 0}
        training_phase_control = TrainingPhaseControl(params)
        outlier_loss_detect = utlis.OutlierLossDetect(30, 3, 3, 10)
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
            # batch_sample is a list of (sample_sequences, clot_voxel_count)

            if model_failed:
                continue

            list_sample_sequence = []
            list_clot_volume = []
            for item in batch_sample:
                list_sample_sequence.append(item[0])
                list_clot_volume.append(item[1])

            list_sample_attention = []
            for idx in range(len(list_clot_volume)):
                if list_clot_volume[idx] <= 0:
                    sample_attention = 1 / np.sqrt(train_loader.min_clot_count)
                else:
                    sample_attention = 1 / np.sqrt(list_clot_volume[idx])
                if list_sample_sequence[idx][0]["whether_important"]:
                    sample_attention = sample_attention * 2.5
                list_sample_attention.append(sample_attention)

            batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4, cube_shape, clot_gt_tensor = \
                utlis.prepare_tensors_pe_transformer(list_sample_sequence, params["embed_dim"], device='cuda:0',
                                                     training_phase=True, get_flatten_vessel_mask=True)

            segmentation_before_softmax = model(
                batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4)
            # [B, 2, N, flatten_dim]

            loss = loss_function.weighted_cross_entropy_loss(segmentation_before_softmax, clot_gt_tensor, class_balance,
                                                             list_sample_attention)
            if i % 10 == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss))

            float_loss = loss.detach().float().cpu().data

            loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss
            if loss_status == "consecutive_outlier":  # this means the model is failed, restart a new one
                del loss, segmentation_before_softmax, clot_gt_tensor
                del batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4, cube_shape

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
                print("size for flatten_vessel_mask", flatten_vessel_mask_deeper_4.size())
                print("size for clot_gt_tensor", clot_gt_tensor.size())
                print("size for segmentation_before_softmax", segmentation_before_softmax.size())
                print("initial class balance:", class_balance)
                print("list_clot_attention:", list_sample_attention)

            del batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask_deeper_4
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
            value_increase = data_dict['value_increase']

            print("back up version has value increase:", value_increase)

            train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                          value_increase, params["voxel_variance"])
            train_loader.prepare_training_dataset()
            test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                         value_increase, params["voxel_variance"])
            test_loader.prepare_testing_dataset()

            model_failed = False  # rolled back

            continue

        recall = num_true_positive / total_clot_voxel
        precision = num_true_positive / (num_true_positive + num_false_positive)

        if recall <= 0 or precision <= 0:
            dice = 0
        else:
            dice = 2 / (1 / recall + 1 / precision)

        loss_ave = loss_ave / accumulative_step / params["batch_ct"]
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
                continue

            if dice_test > 0.5:
                print("updating backup model at dice test:", dice_test)
                save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params,
                                best=False,
                                training_phase_control=training_phase_control,
                                special_name="backup",
                                outlier_loss_detect=outlier_loss_detect,
                                value_increase=value_increase)
            if dice_test < 0.4:
                print("model failed at dice test:", dice_test)
                model_failed = True  # detect failure in the evaluation

            outlier_loss_detect.reset()

            training_phase_control.flip_remaining += 1
            if value_increase[0] >= 0.5:
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.5
                if params["difficulty"] == "decrease":
                    value_increase[0] = value_increase[0] * 1.5
            elif 0.1 <= value_increase[0] < 0.5:
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.1
                if params["difficulty"] == "decrease":
                    value_increase[0] = value_increase[0] * 1.1
            else:
                if params["difficulty"] == "increase":
                    value_increase[0] = value_increase[0] / 1.05
                if params["difficulty"] == "decrease":
                    value_increase[0] = value_increase[0] * 1.05

            value_increase[1] = value_increase[0] * 5

            train_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                          value_increase, params["voxel_variance"])
            train_loader.prepare_training_dataset()
            test_loader.update_clot_simulation_parameter(params["power_range"], params["add_base_range"],
                                                         value_increase, params["voxel_variance"])
            test_loader.prepare_testing_dataset()

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
            list_sample_sequence = []
            list_clot_volume = []
            for item in batch_sample:
                list_sample_sequence.append(item[0])
                list_clot_volume.append(item[1])

            list_sample_attention = []
            for idx in range(len(list_clot_volume)):
                if list_clot_volume[idx] <= 0:
                    sample_attention = 1 / np.sqrt(test_loader.min_clot_count)
                else:
                    sample_attention = 1 / np.sqrt(list_clot_volume[idx])
                if list_sample_sequence[idx][0]["whether_important"]:
                    sample_attention = sample_attention * 2.5
                list_sample_attention.append(sample_attention)

            batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask, cube_shape, clot_gt_tensor = \
                utlis.prepare_tensors_pe_transformer(list_sample_sequence, params["embed_dim"], device='cuda:0',
                                                     training_phase=True, get_flatten_vessel_mask=True)
            segmentation_before_softmax = model(batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask)
            # [B, 2, N, flatten_dim]
            loss = loss_function.weighted_cross_entropy_loss(segmentation_before_softmax, clot_gt_tensor, class_balance,
                                                             list_sample_attention)
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

        loss_ave = loss_ave / len(test_loader) / params["batch_ct"]
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


class TrainingPhaseControl:
    def __init__(self, params):

        self.target_recall = params["target_recall"]
        self.target_precision = params["target_precision"]

        self.flip_recall = params["flip_recall"]
        self.flip_precision = params["flip_precision"]

        self.base_recall = params["base_recall"]
        self.base_precision = params["base_precision"]

        self.current_phase = 'warm_up'
        # 'warm_up', 'recall_phase', 'precision_phase', 'converge_to_recall', 'converge_to_precision'

        self.flip_remaining = params["flip_remaining"]
        # one flip means change the phase 'precision_phase' -> 'recall_phase'

        self.base_relative = params["base_relative"]
        # will not flip util number times recall/precision bigger than precision/recall >= base_relative

        self.max_performance_recall = params["max_performance_recall"]
        self.max_performance_precision = params["max_performance_precision"]
        # force flip when precision/recall > max_performance during precision/recall phase

        self.final_phase = params["final_phase"]  # 'converge_to_recall', 'converge_to_precision'

        self.warm_up_epochs = params["warm_up_epochs"]

        self.previous_phase = None
        self.changed_phase_in_last_epoch = False

        # --------------------------
        # check correctness
        assert 0 <= self.flip_recall <= 1 and 0 <= self.flip_precision <= 1
        assert 0 <= self.base_recall <= 1 and 0 <= self.base_precision <= 1
        assert self.flip_remaining >= 0
        assert self.warm_up_epochs >= 0

        assert self.final_phase in ['converge_to_recall', 'converge_to_precision']
        if self.final_phase == 'converge_to_recall':
            assert 0 < self.target_recall < 1
        if self.final_phase == 'converge_to_precision':
            assert 0 < self.target_precision < 1

        self.precision_to_recall_during_converging = 4
        # the precision and recall will fluctuate around the target performance. When this value to 0, end to training.

        self.epoch_passed = 0
        self.relative_false_positive_penalty = params["initial_relative_false_positive_penalty"]
        # higher means model give less false positives, at the expense of more false negative

        self.history_relative_false_positive_penalty = []
        self.history_recall = []
        self.history_precision = []

    def get_new_relative_false_positive_penalty(self, current_recall, current_precision):
        self._update_history(current_recall, current_precision)
        self.changed_phase_in_last_epoch = self._update_phase(current_recall, current_precision)
        self._update_relative_false_positive_penalty(current_recall, current_precision)
        self.show_status(current_recall, current_precision)
        self.epoch_passed += 1
        return self.relative_false_positive_penalty

    def _update_history(self, current_recall, current_precision):
        self.history_relative_false_positive_penalty.append(self.relative_false_positive_penalty)
        self.history_recall.append(current_recall)
        self.history_precision.append(current_precision)

    def _update_phase(self, current_recall, current_precision):
        # return True for phase change

        if self.previous_phase is None:
            self.previous_phase = self.current_phase  # update previous phase when update current phase

        if self.current_phase == self.final_phase:  # do not update
            return False

        if self.epoch_passed < self.warm_up_epochs:
            self.current_phase = 'warm_up'
            return False

        if self.current_phase == 'warm_up' and self.epoch_passed >= self.warm_up_epochs:
            self.current_phase = 'recall_phase'
            if (current_recall > self.flip_recall and current_recall / (current_precision + 1e-8) > self.base_relative)\
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                self.previous_phase = self.current_phase
                self.current_phase = 'precision_phase'
            print("changing current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
            return True

        if self.current_phase == 'recall_phase':
            if (current_recall > self.flip_recall and current_recall / (current_precision + 1e-8) > self.base_relative)\
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                if self.flip_remaining > 0 or self.final_phase == 'converge_to_precision':
                    self.previous_phase = self.current_phase
                    self.current_phase = 'precision_phase'
                else:
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                print("change current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
                return True

        if self.current_phase == 'precision_phase':
            if (current_precision > self.flip_precision
                and current_precision / (current_recall + 1e-8) > self.base_relative) \
                    or current_recall < self.base_recall or current_precision > self.max_performance_precision:
                if self.flip_remaining > 0:
                    self.previous_phase = self.current_phase
                    self.current_phase = 'recall_phase'
                    self.flip_remaining -= 1
                    print("changing current_phase to:", self.current_phase, 'flip_remaining', self.flip_remaining)
                    return True
                else:
                    assert self.final_phase == 'converge_to_precision'
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                    print("change current_phase to:", self.current_phase)
                    return True
        return False

    def show_status(self, current_recall=None, current_precision=None):
        print("epoch passed:", self.epoch_passed, "current phase:", self.current_phase,
              "relative_false_positive_penalty", self.relative_false_positive_penalty,
              "flip remaining:", self.flip_remaining)
        if current_recall is not None and current_precision is not None:
            print("current (recall, precision)", (current_recall, current_precision))

    def _update_relative_false_positive_penalty(self, current_recall, current_precision):

        if self.current_phase == 'warm_up':
            print("warm_up phase, relative_false_positive_penalty:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'recall_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.15
            print("recall phase, decrease relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'precision_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.13
            print("precision phase, increase relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_recall':

            if current_recall > self.target_recall:  # the recall is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_precision':

            if current_precision > self.target_precision:  # the precision is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty
