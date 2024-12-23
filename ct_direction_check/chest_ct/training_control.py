import sys

sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import torch
import ct_direction_check.training_iter as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

parameters = {

    "mode": 'normal',  # ['temp', 'normal']  temp for debug, normal for training
    "sample_sequence_length": 4000,  # 4000 for high resolution, 1500 for low resolution

    # --------------------------------------------------------
    # training specifics

    "batch_size": 16,

    "device_ids": [0, 0, 1, 2],

    "resume": True,
    "reuse_optimizer": True,
    "reset_best_performance": True,
    "reset_outlier_detect": False,

    "mute_outlier_detect": True,

    "n_epochs": 5000,

    "lr": 1e-3,

    "accumulate_step": 8,

    # -------------------------------------------------------
    # dataloader specifics
    "sample_dir_list":
        ["/data_disk/chest_ct_direction/training_samples/not_clip",
         "/data_disk/chest_ct_direction/training_samples/clip_max_50HU"],

    "sample_interval": (0, 20),
    "shuffle_path_list": True,
    "checkpoint_dir":
        "/data_disk/chest_ct_direction/check_point_different_dataset_size/",

    "saved_model_filename": "model_temp.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": (0, 5),
    "wrong_file_name": None,

    # -------------------------------------------------------
    # model specifics
}

if parameters["mode"] == 'temp':
    parameters["sample_interval"] = (0, 20)

if __name__ == '__main__':
    train_iterations.training(parameters)
