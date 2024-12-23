import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import torch
import pulmonary_embolism_final.training.training_iterations as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

# PID 23340

parameters = {

    "mode": 'normal',  # ['temp', 'normal']  temp for debug, normal for training
    "sample_sequence_length": 4000,  # 4000 for high resolution, 1500 for low resolution

    # --------------------------------------------------------
    # training specifics

    "batch_size_simu": 6,  # number of ct in each batch,
    "batch_size_with_gt": 2,
    "relative_ratio_with_gt": 0.3,  # avoid over-fitting

    "test_id": 0,
    "checkpoint_dir":
        "/data_disk/pulmonary_embolism_final/check_point_dir/use_cta_confirm/high_resolution/"
        "with_annotation_test_id_0/",

    "augment_non_pe": False,
    "augment_pe": False,

    "device_ids": [0, 0, 1, 2],

    "difficulty": "stable",  # in ["increase", "decrease", "stable"]

    "resume": True,
    "reuse_phase_control": True,
    "reuse_optimizer": False,
    "reset_best_performance": True,
    "reset_outlier_detect": True,

    "n_epochs": 5000,

    "lr": 1e-3,

    "accumulate_step": 36,  # real batch size is (batch_simu + batch_with_gt) * accumulate_step

    "num_prepared_dataset_test": 1,  # number of prepared dataset during testing, higher the more accuracy, but slower

    "target_recall": 0.2,
    "target_precision": None,
    "flip_recall": 0.2,  # reach this recall, then can flip
    "flip_precision": 0.2,  # reach this precision, then can flip
    "base_relative": 1.01,  # 1.01
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.8,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.8,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance

    "flip_remaining": 2,
    "converge_to_final_phase": False,

    "final_phase": 'converge_to_recall',
    "warm_up_epochs": 2,
    "initial_relative_false_positive_penalty": 0.2,

    # -------------------------------------------------------
    # dataloader specifics
    'num_workers_simu': 24,  # num CPU for the parallel data loading, if you see load average < num_workers, reduce this
    'num_workers_with_gt': 6,

    "sample_dir_list_non_pe":
        ["/data_disk/pulmonary_embolism_final/training_samples_simulate_clot/"
         "high_resolution/not_pe_ready_not_denoise", ],

    "sample_dir_list_pe":
        ['/data_disk/pulmonary_embolism_final/training_samples_with_annotation_vessel_high_recall_cta_confirm/'
         'high_resolution/pe_ready_not_denoise', ],

    "sample_interval": (0, 1),
    "shuffle_path_list": True,

    "saved_model_filename": "model_guided.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    "wrong_file_name": None,

    # -------------------------------------------------------
    # clot simulation specifics
    "top_dict_clot_pickle":
        '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/merged_clot_seeds/shuffled_complete.pickle',
    # '/data_disk/pulmonary_embolism/simulated_lesions/clot_sample_list_reduced/volume_range_25%/',

    "clot_volume_range": (500, 30000),  # the raw volume of the clot seed
    "num_clot_each_sample_range": (2, 2),  # number of clots applied to each sample sequence

    "value_increase": None,
    # very important parameter, larger the easier.
    # start with large value like (1, 2), for which the model can get very high performance, then gradually decrease it

    "min_clot": 500,  # the volume of clot when applied to vessel
    "power_range": (-0.3, 0.6),
    "add_base_range": (0, 3),
    "voxel_variance": (0.99, 1),

    # -------------------------------------------------------
    # model specifics
    "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
    "in_channel": 1,  # 1 for CT
    "cnn_features": 128,  # number of cnn kernels
    "given_features": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
    "embed_dim": 192,  # the embedding dimension for each input cubes.
    # Require: embed_dim % int(8 * encoder_heads) == 0
    "num_heads": 12,
    "encoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "feature_vector"
    "interaction_depth": 2,
    "decoding_depth": 2,  # encoding blocks are transformer blocks that are guided by "flatten_blood_vessel_mask"
    "segmentation_depth": 1,
    "mlp_ratio": 2.0,  # the DNN inside transformer blocks: len(vector) -> int(mlp_ratio * len(vector)) -> len(vector)
}

if parameters["mode"] == 'temp':
    parameters["sample_interval"] = (0, 20)

if __name__ == '__main__':
    train_iterations.training(parameters)
