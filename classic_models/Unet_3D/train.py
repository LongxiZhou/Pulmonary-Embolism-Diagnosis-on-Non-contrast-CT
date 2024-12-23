import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms
import collections
import sys
import classic_models.Unet_3D.U_net_Model_3D as unm
import classic_models.Unet_3D.loss_functions as unlf
from classic_models.Unet_3D.dataset import RandomFlipWithWeight, RandomRotateWithWeight, ToTensorWithWeight, \
    SwapAxisWithWeight, WeightedTissueDataset3D
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')


parameters = {

}


def save_checkpoint(epoch, model, optimizer, history, best_eval, params=None, best=True, phase_info=None):
    # phase_info = [current_phase, current_base, flip_remaining, fluctuate_epoch],
    # current_phase in {"recall", "precision", "fluctuate"}
    # best eval means: nodule level recall > target_performance (like 0.8), and with smallest false positive cubes
    # best eval: (nodule level recall, num_false_positive_cubes)
    if params is None:
        params = parameters
    filename = params["saved_model_filename"]
    if not best:  # this means we store the current model_guided
        filename = "current_" + params["saved_model_filename"]
    filename = os.path.join(params["checkpoint_dir"], filename)
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_eval': best_eval,
        'phase_info': phase_info,
    }, filename)


def train_loop(model, optimizer, train_loader, test_loader, params=None, resume=True):
    if params is None:
        params = parameters
    if resume and params["best_eval"] is not None:  # direct give a best_f1
        best_eval = params["best_eval"]
    else:
        best_eval = [0, np.Inf, 0]
    saved_model_path = os.path.join(params["checkpoint_dir"], 'current_' + params["saved_model_filename"])
    if resume and os.path.isfile(saved_model_path):
        data_dict = torch.load(saved_model_path)
        epoch_start = data_dict["epoch"]
        if type(model) == nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        optimizer.load_state_dict(data_dict["optimizer"])
        history = data_dict["history"]
        phase_info = data_dict["phase_info"]
        best_eval = data_dict["best_eval"]
        print("best_eval is:", best_eval)
    else:  # this means we do not have a checkpoint
        epoch_start = 0
        history = collections.defaultdict(list)
        phase_info = None

    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

    if phase_info is None:
        base = 1
        precision_phase = True  # if in precision phase, the model_guided increase the precision
        flip_remaining = params["flip_remaining:"]  # initially, model_guided has high recall low precision. Thus, we initially
        # in the precision phase. When precision is high and recall is low, we flip to recall phase.
        # flip_remaining is the number times flip precision phase to recall phase

        fluctuate_phase = False
        # when flip_remaining is 0 and the model_guided reached target_performance during recall phase, change to fluctuate
        #  phase during this phase, the recall fluctuate around the target_performance.

        fluctuate_epoch = 0

        current_phase = "precision"
    else:
        current_phase, base, flip_remaining, fluctuate_epoch = phase_info
        if current_phase == 'precision':
            precision_phase = True
        else:
            precision_phase = False
        if current_phase == 'fluctuate':
            fluctuate_phase = True
        else:
            fluctuate_phase = False
    print("current_phase, base, flip_remaining, fluctuate_epoch:", current_phase, base, flip_remaining, fluctuate_epoch)

    previous_recall = 0
    precision_to_recall_count = 4

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % (epoch))
        print("fluctuate_epoch:", fluctuate_epoch)
        if fluctuate_epoch > 50:
            break
        if precision_to_recall_count < 0:
            break
        model.train()
        for i, sample in enumerate(train_loader):

            image = sample["image"].to(params["device"]).float()
            label = sample["label"].to(params["device"]).float()
            weight = sample["weight"].to(params["device"]).float()
            # weight[:, 0, :, :, :] for false positive penalty, weight[:, 1, :, :, :] for false negative prediction
            pred = model(image)
            maximum_balance = params["balance_weights"][0]
            hyper_balance = base
            if hyper_balance > maximum_balance:
                hyper_balance = maximum_balance

            multiplier = 0.1

            loss = unlf.cross_entropy_pixel_wise_multi_class_3d(pred, label, weight,
                                                                (hyper_balance, multiplier/hyper_balance))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\tstep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss), base,
                  (hyper_balance, multiplier / hyper_balance))
        print("\tEvaluating")

        eval_vals_train = evaluate(model, test_loader, params)
        print("evaluation results:\n", eval_vals_train)

        print("flip_remaining:", flip_remaining, "precision_phase:", precision_phase)

        if epoch >= 5 and not fluctuate_phase:  # recall will be very very high, start with precision phase

            if eval_vals_train["sensitivity"] > params["baseline_sensitivity"] and precision_phase:
                print("precision phase, increase base to", base * 1.13)
                base = base * 1.13
            elif precision_phase:
                precision_phase = False
                print("change to recall phase")

            if eval_vals_train["sensitivity"] < params["target_sensitivity"] and not precision_phase:
                print("recall phase, decrease base to", base / 1.15)
                base = base / 1.15

            elif not precision_phase:
                if flip_remaining > 0:
                    flip_remaining -= 1
                    precision_phase = True
                    print("change to precision phase")
                else:
                    print("change to fluctuate phase")
                    fluctuate_phase = True

        if fluctuate_phase:
            if eval_vals_train["sensitivity"] > params["target_sensitivity"] > previous_recall:
                precision_to_recall_count -= 1
                print("precision to recall count:", precision_to_recall_count)
            if eval_vals_train["sensitivity"] < params["target_sensitivity"]:
                print("fluctuate phase, decrease base to", base / 1.025)
                base = base / 1.025
            else:
                print("fluctuate phase, increase base to", base * 1.024)
                base = base * 1.024
            fluctuate_epoch += 1
            previous_recall = eval_vals_train["sensitivity"]

        if precision_phase:
            current_phase = 'precision'
        elif fluctuate_phase:
            current_phase = 'fluctuate'
        else:
            current_phase = 'recall'

        phase_info = [current_phase, base, flip_remaining, fluctuate_epoch]

        for k, v in eval_vals_train.items():
            history[k + "_train"].append(v)

        if eval_vals_train["num_false_nodules"] < best_eval[1] and eval_vals_train["sensitivity"] > 0.9:
            print("saving model_guided as:", str(params["test_id"]) + "_saved_model.pth")
            best_eval = [eval_vals_train["nodule_recall"], eval_vals_train["num_false_nodules"],
                         eval_vals_train["sensitivity"]]
            save_checkpoint(epoch, model, optimizer, history, best_eval, params, phase_info=phase_info)
        save_checkpoint(epoch, model, optimizer, history, best_eval, params, False, phase_info=phase_info)
    print("Training finished")
    print("best_eval:", best_eval)


def evaluate(model, test_loader, params):
    model.eval()
    with torch.no_grad():

        weighted_positive_rate = 0  # average regions detected for each nodule (volume balanced)
        num_false_positive_cube = 0  # how many negative tubes false considered as nodule?

        num_tube_with_nodule = 0
        num_tube_without_nodule = 0

        num_tube_true_positive = 0  # tube contains nodule, and model_guided discovered it

        for i, sample in enumerate(test_loader):
            # batch size is 1

            contain_nodule = False

            image = sample["image"].to(params["device"]).float()
            label = sample["label"][:, 1:, :, :, :].to(params["device"]).float()

            total_positive_gt = label.sum().float().item()

            if 0 < total_positive_gt <= 27:
                continue

            if total_positive_gt > 27:
                contain_nodule = True
                num_tube_with_nodule += 1
            else:
                num_tube_without_nodule += 1

            pred = model(image)

            pred = (pred[:, 1, :, :, :] > pred[:, 0, :, :, :]).float().unsqueeze(1)

            if contain_nodule:
                pred_tp = pred * label
                tp = pred_tp.sum().float().item()
                weighted_positive_rate += tp / total_positive_gt
                if tp > 27:
                    num_tube_true_positive += 1
            else:
                if pred.sum().float().item() > 64:  # 3 mm^3
                    num_false_positive_cube += 1

        weighted_positive_rate = weighted_positive_rate / num_tube_with_nodule

        return_dict = {"nodule_recall": weighted_positive_rate, "num_tube_with_nodule": num_tube_with_nodule,
                       "true positive nodule": num_tube_true_positive,
                       "sensitivity": num_tube_true_positive / num_tube_with_nodule,
                       "num_false_nodules": num_false_positive_cube,
                       "num_sample_without_nodule": num_tube_without_nodule}

        return return_dict


def predict(model, test_loader, params):
    model.eval()
    prediction_list = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample["image"].to(params["device"]).float()
            pred = model(image)
            pred = pred.cpu().numpy()
            prediction_list.append(pred)
        predictions = np.concatenate(prediction_list, axis=0)
        return predictions


def training(parameter):
    params = parameter
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    train_transform = torchvision.transforms.Compose([
        ToTensorWithWeight(),
        RandomFlipWithWeight(),
        RandomRotateWithWeight(),
        SwapAxisWithWeight()
    ])
    test_transform = torchvision.transforms.Compose([
        ToTensorWithWeight()
    ])

    train_dataset = WeightedTissueDataset3D(
        parameter["train_data_dir"],
        transform=train_transform,
        channels_data=parameter["channels_data"],
        channels_weight=parameter["channels_weight"],
        mode="train",
        test_id=parameter["test_id"],
    )

    test_dataset = WeightedTissueDataset3D(
        parameter["train_data_dir"],
        transform=test_transform,
        channels_data=parameter["channels_data"],
        channels_weight=parameter["channels_weight"],
        mode='test',
        test_id=parameter["test_id"],
    )

    print("train:", parameter["train_data_dir"], len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,
                                               num_workers=params["workers"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=params["workers"])

    if params["encoders"] == 4:
        model = unm.UNet3D(in_channels=params["channels_data"], out_channels=2, init_features=params["init_features"])
    elif params["encoders"] == 3:
        model = unm.UNet3DSimple(in_channels=params["channels_data"], out_channels=2,
                                 init_features=params["init_features"])
    elif params["encoders"] == 2:
        model = unm.UNet3DSimplest(in_channels=params["channels_data"], out_channels=2,
                                   init_features=params["init_features"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    train_loop(model, optimizer, train_loader, test_loader, params)
