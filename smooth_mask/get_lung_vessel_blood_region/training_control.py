import torch
import smooth_mask.get_lung_vessel_blood_region.training_iteration as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {
    # --------------------------------------------------------
    # training specifics
    "resume": True,
    "reuse_phase_control": False,
    "reset_best_performance": False,
    "reset_outlier_detect": True,

    "random_augment": True,  # random flip, rotate and swap axis

    "n_epochs": 5000,
    "save_interval": 1,  # checkpoints other than best model, current model

    "lr": 1e-3,
    "batch_size": 2,  # number of ct in each batch
    "accumulate_step": 2,  # real batch size is batch_ct * accumulate_step
    "shuffle": True,  # shuffle the before each training iteration.
    "drop_last": True,  # whether drop last incomplete batch during training
    "num_test_replicate": 2,  # the test is prepared before training, more test will have more lesion variants

    # batch_normalize
    "normalize_by_positive_voxel": False,  # when batch is small, num positive voxel can variant very large.
                                           # set True to normalize the batch loss by num_positive_voxels.
    "normalize_protocol": 'sqrt',  # in ['abs', 'sqrt', 'log']
    "normalize_base": 100,  # loss = loss / normalize_protocol(num_positive_voxel + normalize_base)

    # phase information
    "target_recall": None,
    "target_precision": 0.825,
    "flip_recall": 0.8,  # reach this recall, then can flip
    "flip_precision": 0.85,  # reach this precision, then can flip
    "base_relative": 1.,
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.9,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.9,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance
    "flip_remaining": 0,
    "final_phase": 'converge_to_precision',
    "warm_up_epochs": 0,
    "initial_relative_false_positive_penalty": 21.162910172734502,

    # -------------------------------------------------------
    # model specifics
    "model_size": 'small',  # in ['small', 'median', 'large']
    "in_channels": 1,  # 1 for CT
    "out_channels": 2,  # channel 0 for positive, 1 for negative
    "init_features": 16,

    "checkpoint_dir":
        "/data_disk/artery_vein_project/extract_blood_region/check_points/256_final_augment/",

    "saved_model_filename": "model_smooth.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # -------------------------------------------------------
    # dataset and dataloader specifics
    'num_workers': 4,  # num CPU for the parallel data loading
    "sample_interval": (0, 1),  # sample_path_list = sample_path_list[sample_interval[0]:: sample_interval[1]]

    "list_train_data_dir": ['/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/CTA/stack_array_artery',
                            '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/CTA/stack_array_vein',
                            '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/CTA/stack_array_blood',
                            '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/non-contrast/stack_array_artery',
                            '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/non-contrast/stack_array_vein',
                            '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/'
                            '256_v1/non-contrast/stack_array_blood'
                            ],

    "list_test_data_dir":  None,
    "test_id": 0,
    "wrong_file_name": None,

    # -------------------------------------------------------
    # lesion simulation specifics
    "difficulty_level": 6,  # the difficulty for simulated lesions, in range [0, 8], see "SimulateLesionDataset"
    "penalty_range": (0.05, 0.5),  # penalty for simulated lesions. Ave penalty for non-lesion ranges from 0.15 to 0.5
    "lesion_top_dict": '/data_disk/artery_vein_project/extract_blood_region/lesion_simulation',
    "num_lesion_applied": (0, 3),

    # -------------------------------------------------------
    # visualization specifics
    "visualize_interval": 1
}

if parameters["list_test_data_dir"] is None:
    parameters["list_test_data_dir"] = parameters["list_train_data_dir"]


if __name__ == '__main__':
    train_iterations.training(parameters)
