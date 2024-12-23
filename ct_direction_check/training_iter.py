import torch
import os
from ct_direction_check.model_cnn import AlexNet
import ct_direction_check.chest_ct.dataset_and_dataloader as dataset
from pulmonary_embolism_final.utlis.phase_control_and_outlier_loss_detect import \
    OutlierLossDetect
import collections
import torch.nn as nn
import numpy as np


def cross_entropy_loss(prediction_before_softmax, ground_truth):
    """
    all_file parameters should on GPU, with float32 data type.
    :param prediction_before_softmax: [batch_size, class_num], NOT soft_maxed!
    :param ground_truth: [batch_size, class_num], each pixel with value [0, 1]
    :return: a float with value [0, inf)
    """

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_prediction_probability = -softmax_then_log(prediction_before_softmax)

    return_tensor = log_prediction_probability * ground_truth
    loss = torch.sum(return_tensor)

    return loss


def training(params):
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    model = AlexNet()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=params["device_ids"])
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    dataset_train = dataset.SampleDataset(
        sample_dir_list=params["sample_dir_list"], sample_interval=params["sample_interval"],
        batch_size=params["batch_size"], mode='train', test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"], drop_last=True)

    dataset_test = dataset.SampleDataset(
        sample_dir_list=params["sample_dir_list"], sample_interval=params["sample_interval"],
        batch_size=params["batch_size"], mode='test', test_id=params["test_id"],
        wrong_file_name=params["wrong_file_name"], drop_last=False)

    print("there are:", len(dataset_train), "training samples")
    print("there are:", len(dataset_test), "testing samples")

    train_loop(model, optimizer, dataset_train, dataset_test, params)


def save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params=None, best=True,
                    special_name=None, outlier_loss_detect=None):
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
        'outlier_loss_detect': outlier_loss_detect,
        'current_performance': current_performance,
    }, save_path, _use_new_zipfile_serialization=False)


def train_loop(model, optimizer, dataset_train, dataset_test, params):
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

        outlier_loss_detect = data_dict['outlier_loss_detect']
        if outlier_loss_detect is None or params["reset_outlier_detect"]:
            outlier_loss_detect = OutlierLossDetect(max(300, int(3 * params["accumulate_step"])), 3, 3, 30,
                                                    mute=params["mute_outlier_detect"])
    else:
        epoch_start = 0
        history = collections.defaultdict(list)
        best_performance = {"loss_ave_test": np.inf, "accuracy_test": 0}
        outlier_loss_detect = OutlierLossDetect(max(300, int(3 * params["accumulate_step"])), 3, 3, 30,
                                                mute=params["mute_outlier_detect"])

    if params["reset_best_performance"]:
        best_performance = {"loss_ave_test": np.inf, "accuracy_test": 0}

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % epoch)
        loss_ave = 0

        num_true_prediction = 0
        num_false_prediction = 0

        accumulative_step = 0

        model.train()
        for i, batch_sample in enumerate(dataset_train):
            # batch_sample is (input_tensor, ground_truth_tensor)

            input_tensor, ground_truth_tensor = batch_sample

            input_tensor = input_tensor.cuda(device='cuda:0')
            ground_truth_tensor = ground_truth_tensor.cuda(device='cuda:0')

            prediction_before_softmax = model(input_tensor)
            # [B, 48]

            loss = cross_entropy_loss(prediction_before_softmax, ground_truth_tensor)
            float_loss = loss.detach().float().cpu().data
            if i % 10 == 0:
                print("\tStep [%d/%d], loss=%.4f" % (i + 1, int(len(dataset_train) / params["batch_size"]), float_loss))

            loss_status = outlier_loss_detect.update_new_loss(float_loss)  # True for good loss, False for bad loss

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

            prediction_before_softmax = prediction_before_softmax.detach().cpu().numpy()  # [B, 48]

            predicted_class = prediction_before_softmax.argmax(axis=1)  # [B, ]
            ground_truth_array = ground_truth_tensor.detach().cpu().numpy().argmax(axis=1)

            for index in range(len(predicted_class)):
                if predicted_class[index] == ground_truth_array[index]:
                    num_true_prediction += 1
                else:
                    num_false_prediction += 1

        loss_ave = loss_ave / accumulative_step / params["batch_size"]

        accuracy = num_true_prediction / (num_false_prediction + num_true_prediction)

        print("\nloss average on each image train:", loss_ave)
        print("accuracy train:", accuracy, 'or', num_true_prediction, '/', num_false_prediction + num_true_prediction)
        history["loss_average_on_each_image_training"].append(loss_ave)
        history["accuracy_for_each_training_epoch"].append(accuracy)

        print("\tEvaluating")

        loss_ave_test, accuracy_test = evaluate(model, dataset_test, params, history)

        current_performance = {"loss_ave_train": loss_ave, "loss_ave_test": loss_ave_test,
                               "accuracy_train": accuracy, "accuracy_test": accuracy_test}

        if current_performance["accuracy_test"] > best_performance["accuracy_test"]:
            print("\nNew best model_guided at accuracy test:", current_performance["accuracy_test"], '\n')
            best_performance = current_performance
            save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=True,
                            outlier_loss_detect=outlier_loss_detect)

        save_checkpoint(epoch, model, optimizer, history, best_performance, current_performance, params, best=False,
                        outlier_loss_detect=outlier_loss_detect)


def evaluate(model, dataset_test, params, history):
    loss_ave = 0
    num_true_prediction = 0
    num_false_prediction = 0

    accumulative_step = 0

    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(dataset_test):
            # batch_sample is (input_tensor, ground_truth_tensor)

            input_tensor, ground_truth_tensor = batch_sample

            input_tensor = input_tensor.cuda(device='cuda:0')
            ground_truth_tensor = ground_truth_tensor.cuda(device='cuda:0')

            prediction_before_softmax = model(input_tensor)
            # [B, 48]

            loss = cross_entropy_loss(prediction_before_softmax, ground_truth_tensor)
            float_loss = loss.detach().float().cpu().data
            loss_ave += float_loss

            prediction_before_softmax = prediction_before_softmax.detach().cpu().numpy()  # [B, 48]

            predicted_class = prediction_before_softmax.argmax(axis=1)  # [B, ]
            ground_truth_array = ground_truth_tensor.detach().cpu().numpy().argmax(axis=1)

            for index in range(len(predicted_class)):
                if predicted_class[index] == ground_truth_array[index]:
                    num_true_prediction += 1
                else:
                    num_false_prediction += 1

            accumulative_step += 1

        loss_ave = loss_ave / accumulative_step / params["batch_size"]
        accuracy = num_true_prediction / (num_false_prediction + num_true_prediction)

        print("\nloss average on each image test:", loss_ave)
        print("accuracy test:", accuracy, 'or', num_true_prediction, '/', num_false_prediction + num_true_prediction)
        history["loss_average_on_each_image_testing"].append(loss_ave)
        history["accuracy_for_each_testing_epoch"].append(accuracy)

        return loss_ave, accuracy
