import torch
import pulmonary_embolism_v2.transformer_PE_4D.training_iteration as train_iterations
import os
import Tool_Functions.Functions as Functions

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {
    # --------------------------------------------------------
    # training specifics

    "difficulty": "increase",  # in ["increase", "decrease", "stable"]

    "resume": True,
    "reuse_phase_control": True,
    "reset_best_performance": False,
    "reset_outlier_detect": False,

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_ct": 6,  # number of ct in each batch,    2 for lung region, 2 for all_file ct,
    "batch_ct_test": 4,
    "accumulate_step": 32,  # real batch size is batch_ct * accumulate_step

    # prepare dataset (pick samples from lists and form tensors) can be time consuming
    "num_prepared_dataset_train": 2,  # number of prepared dataset during training
    "num_prepared_dataset_test": 1,  # number of prepared dataset during testing, higher the more accuracy, but slower
    "reuse_count": 3,  # number of times each training dataset will be used

    "target_recall": 0.6,
    "target_precision": None,
    "flip_recall": 0.6,  # reach this recall, then can flip
    "flip_precision": 0.6,  # reach this precision, then can flip
    "base_relative": 1.03,
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.85,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.85,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance

    "flip_remaining": 2,

    "final_phase": 'converge_to_recall',
    "warm_up_epochs": 3,
    "initial_relative_false_positive_penalty": 1.,

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

    # -------------------------------------------------------
    # dataloader specifics
    'num_workers': 1,  # num CPU for the parallel data loading, if you see load average < num_workers, reduce this
    # because establishing new thread is very costly

    "train_data_dir":
        "/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/"
        "merged-refine_length-3000_branch-7/",
    "test_data_dir":
        "/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/"
        "merged-refine_length-3000_branch-7/",

    "checkpoint_dir":
        "/home/zhoul0a/Desktop/pulmonary_embolism/check_points/Simulate_Clot/high_variance_clot/decrease_3 (v6)/",

    "saved_model_filename": "model_guided.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_file_name": Functions.pickle_load_object('/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/'
                                                    'simulate_clot/training_dataset/CTA_name_set.pickle'),

    "importance_file_name": Functions.pickle_load_object('/home/zhoul0a/Desktop/pulmonary_embolism/'
                                                         'sample_sequence_dataset/simulate_clot/training_dataset/'
                                                         'important_name_list.pickle'),

    # -------------------------------------------------------
    # clot simulation specifics
    "min_clot_count": 500,

    "power_range": (-0.3, 0.6),  # (-0.3, 0.6)
    "add_base_range": (0, 3),

    "value_increase": (0.03, 0.15),
    # very important parameter, larger the easier.
    # start with large value like (1, 2), for which the model can get very high performance, then gradually decrease it

    "voxel_variance": (0.99, 1),  # (0.5, 1)

    "list-clot_sample_dict_dir":
        '/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/merged_clot_seeds/shuffled_complete.pickle'
}


if __name__ == '__main__':
    train_iterations.training(parameters)
