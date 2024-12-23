import torch
import registration_pulmonary.training.training_iterations as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {

    "mode": 'normal',  # ['debug', 'normal']

    # --------------------------------------------------------
    # training specifics
    "resume": True,
    "reuse_phase_control": True,
    "reuse_optimizer": False,
    "reset_best_performance": False,
    "reset_outlier_detect": True,
    "augment": True,

    "use_penalty_weight": False,
    "use_outlier_loss_detect": False,
    "use_flow_based_loss": True,

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_size_train": 4,  # number of ct in each batch,    2 for lung region, 2 for all_file ct,
    "batch_size_test": 4,
    "accumulate_step": 1,  # real batch size is batch_ct * accumulate_step

    "target_rough": 2.5,
    "flip_high_rough": 7,  # reach this roughness, then can flip
    "flip_low_rough": 2,  # reach this roughness, then can flip
    "flip_remaining": 3,

    "warm_up_epochs": 5,
    "relative_penalty_for_flow": 1.,

    "ncc_window_length": 9,
    "ncc_stride": 3,
    "precision_for_jacobi": 4,  # in [1, 2, 4], the jacobi determinant with error of h^(precision_for_jacobi)
    "weight_for_important": 5,  # number of times more focus on these samples

    # -------------------------------------------------------
    # augmentation specifics
    "ratio_easy": 1.0,  # easy means moving image is same with the fixed image
    "random_translate_voxel_range": (-3, 3),  # for each dim, translate by random.randint(a, b)

    # -------------------------------------------------------
    # dataloader specifics
    "sample_dir_list": ["/data_disk/pulmonary_registration/cast_CTA_to_CT/training_sample_32"],

    "sample_interval": (0, 1),
    "checkpoint_dir":
        "/data_disk/pulmonary_registration/check_points/registration_32/",

    "saved_model_filename": "model_registration.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_file_name": None,
    "important_file_name": None,

    # -------------------------------------------------------
    # model specifics
    "image_length": 32,  # shape of the image tube
    "split_positive_and_negative": False,  # whether use independent models for positive flow and negative flow
    "num_landmark": 24,  # number of landmark the model try to extract for registration
    "depth_get_landmark": 3,  # number of MaxPooling of U-net to extract landmark for registration
    "depth_refine_flow": 3,  # number of MaxPooling of U-net to form registration flow from landmarks

}

if parameters["mode"] == 'debug':
    parameters["sample_interval"] = (0, 20)

if __name__ == '__main__':
    train_iterations.training(parameters)
