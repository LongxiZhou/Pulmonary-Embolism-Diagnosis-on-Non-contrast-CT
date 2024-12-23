import torch
import registration_pulmonary.training_v3.training_iterations as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {

    "mode": 'normal',  # ['debug', 'normal']
    "phase_shift": False,  # whether shift between smooth flow phase and accurate register phase

    # --------------------------------------------------------
    # training specifics

    "resume": True,
    "reuse_phase_control": True,
    "reuse_optimizer": True,
    "reset_best_performance": False,
    "reset_outlier_detect": True,

    "use_penalty_weight": False,
    "use_outlier_loss_detect": False,

    "use_flow_based_loss": True,
    "include_negative_jacobi_loss": False,
    "tension_loss_augment_ratio": 1,
    # True, flow_based_loss = negative_jacobi_loss + flow_tension_loss + flow_gradient_loss

    "use_ncc_loss": False,
    "use_mae_normalized_loss": True,
    "use_mse_normalized_loss": False,

    "ratio_ncc_mae_mse": [1, 1, 1],

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_size_train": 2,  # number of ct in each batch, please % 2 == 0
    "batch_size_test": 2,
    "accumulate_step": 8,  # real batch size is batch_ct * accumulate_step

    "target_rough": 2.5,
    "flip_high_rough": 7,  # reach this roughness, then can flip
    "flip_low_rough": 2,  # reach this roughness, then can flip
    "flip_remaining": 3,

    "warm_up_epochs": 5,
    "relative_penalty_for_flow": 1.,

    "ncc_window_length": 9,
    "ncc_stride": 3,
    "precision_for_jacobi": 1,  # in [1, 2, 4], the jacobi determinant with error of h^(precision_for_jacobi)
    "weight_for_important": 5,  # number of times more focus on these samples

    "channel_weight": [1, 5, 2, 20],  # [image, vessel_depth_normalized, vessel_branch, vessel_mask]

    # -------------------------------------------------------
    # augmentation specifics
    "augment": True,  # each sample apply random swap_axis, rotate and flip
    "ratio_swap": 0.8,  # split paired CTA and non-contrast:
    "ratio_non_to_non": 0.25,  # register non-contrast to another non-contrast
    "ratio_same_to_same": 0.,  # moving and fixed are same image (may translate a little)
    "ratio_apply_translate": 0.,
    "random_translate_voxel_range": (-3, 3),  # for each dim, translate by random.randint(a, b)

    # -------------------------------------------------------
    # dataloader specifics
    "sample_dir_list": ["/data_disk/pulmonary_registration/cast_CTA_to_CT_v3/training_sample_256"],

    "sample_interval": (0, 1),
    "checkpoint_dir":
        "/data_disk/pulmonary_registration/check_points_v3/registration_256/",

    "saved_model_filename": "model_registration.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_file_name": None,
    "important_file_name": None,

    # -------------------------------------------------------
    # model specifics
    "image_length": 256,  # shape of the image tube
    "split_positive_and_negative": False,  # whether use independent models for positive flow and negative flow
    "num_landmark": 18,  # number of landmark the model try to extract for registration
    "depth_get_landmark": 0,  # number of MaxPooling of U-net to extract landmark for registration
    "depth_refine_flow": 3,  # number of MaxPooling of U-net to form registration flow from landmarks

    # -------------------------------------------------------
    # other settings
    "visualize_interval": 10
}

if parameters["mode"] == 'debug':
    parameters["sample_interval"] = (0, 20)

if __name__ == '__main__':
    train_iterations.training(parameters)
