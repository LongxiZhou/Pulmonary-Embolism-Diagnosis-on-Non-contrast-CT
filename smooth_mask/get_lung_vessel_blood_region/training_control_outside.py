import sys
sys.path.append('/home/chuy/longxi_training/longxi_platform')
import torch
import smooth_mask.get_lung_vessel_blood_region.training_iteration as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parameters = {
    # --------------------------------------------------------
    # training specifics
    "resume": True,
    "reuse_phase_control": True,
    "reset_best_performance": False,
    "reset_outlier_detect": False,

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_size": 2,  # number of ct in each batch
    "accumulate_step": 1,  # real batch size is batch_ct * accumulate_step
    "shuffle": True,  # shuffle the before each training iteration.
    "drop_last": True,  # whether drop last incomplete batch during training
    "num_test_replicate": 2,  # the test is prepared before training, more test will have more lesion variants

    # phase information
    "target_recall": 0.875,
    "target_precision": None,
    "flip_recall": 0.9,  # reach this recall, then can flip
    "flip_precision": 0.9,  # reach this precision, then can flip
    "base_relative": 1.,
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.91,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.91,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance
    "flip_remaining": 2,
    "final_phase": 'converge_to_recall',
    "warm_up_epochs": 3,
    "initial_relative_false_positive_penalty": 10.,

    # -------------------------------------------------------
    # model specifics
    "model_size": 'small',  # in ['small', 'median', 'large']
    "in_channels": 1,  # 1 for CT
    "out_channels": 2,  # channel 0 for positive, 1 for negative
    "init_features": 16,

    "checkpoint_dir":
        "/home/chuy/Desktop/longxi_training/extract_blood_region/check_points/",

    "saved_model_filename": "model_smooth.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # -------------------------------------------------------
    # dataset and dataloader specifics
    'num_workers': 4,  # num CPU for the parallel data loading
    "sample_interval": (0, 1),  # sample_path_list = sample_path_list[sample_interval[0]:: sample_interval[1]]

    "list_train_data_dir": ['/home/chuy/Desktop/longxi_training/extract_blood_region/training_samples/'
                            '256_v1/CTA/stack_array_artery',
                            '/home/chuy/Desktop/longxi_training/extract_blood_region/training_samples/'
                            '256_v1/CTA/stack_array_vein',
                            '/home/chuy/Desktop/longxi_training/extract_blood_region/training_samples/'
                            '256_v1/non-contrast/stack_array_artery',
                            '/home/chuy/Desktop/longxi_training/extract_blood_region/training_samples/'
                            '256_v1/non-contrast/stack_array_vein'],

    "list_test_data_dir":  None,
    "test_id": 0,
    "wrong_file_name": None,

    # -------------------------------------------------------
    # lesion simulation specifics
    "difficulty_level": 8,  # the difficulty for simulated lesions, in range [0, 8], see "SimulateLesionDataset"
    "penalty_range": (0.05, 0.5),  # penalty for simulated lesions. Ave penalty for non-lesion ranges from 0.15 to 0.5
    "lesion_top_dict": '/data_disk/artery_vein_project/extract_blood_region/lesion_simulation',
    "num_lesion_applied": 4,

    # -------------------------------------------------------
    # visualization specifics
    "visualize_interval": 1
}

if parameters["list_test_data_dir"] is None:
    parameters["list_test_data_dir"] = parameters["list_train_data_dir"]


if __name__ == '__main__':
    train_iterations.training(parameters)
