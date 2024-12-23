import torch
import segment_clot_cta.training_pe_v3_version2.training_iterations as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {

    "mode": 'normal',  # ['temp', 'normal']  temp for debug, normal for training

    # --------------------------------------------------------
    # training specifics
    "difficulty": "stable",  # in ["increase", "decrease", "stable"]
    # decrease & increase apply to global bias of simulated clot

    "resume": True,
    "reuse_phase_control": True,
    "reuse_optimizer": False,
    "reset_best_performance": False,
    "reset_outlier_detect": True,
    "augment": True,
    "trace_clot": True,
    "roi": "blood_vessel",  # region for search clot, in ["blood_vessel", "blood_region"],
    # but the clot simulation always in "blood_region"

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_size": 6,  # number of ct in each batch,    2 for lung region, 2 for all_file ct,
    "batch_size_test": 6,
    "accumulate_step": 8,  # warm-up usually from 8. Lower the faster, but more risk of model failure
                            # real batch size is batch_ct * accumulate_step

    "num_prepared_dataset_test": 2,  # number of prepared dataset during testing, higher the more accuracy, but slower

    "target_recall": 0.8,
    "target_precision": None,
    "flip_recall": 0.8,  # reach this recall, then can flip
    "flip_precision": 0.8,  # reach this precision, then can flip
    "base_relative": 1.01,  # 1.01
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.85,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.85,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance

    'min_dice_backup': 0.5,  # reach this dice, model may be saved as backup
    'min_dice': 0.03,  # reach this dice, model is considered as failed and will roll back to backup
    'min_dice_at_flip': 0.4,  # reach this dice at flip (recall == precision),
                              # model is considered as failed and will roll back to backup

    'min_dice_less_than_best': 0.2,  # if at flip, the dice is less than this amount to best model, then roll back

    "flip_remaining": 2,

    "final_phase": 'converge_to_recall',
    "warm_up_epochs": 1,
    "initial_relative_false_positive_penalty": 5.,

    # -------------------------------------------------------
    # dataloader specifics
    'num_workers': 32,  # num CPU for the parallel data loading, if you see load average < num_workers, reduce this

    # simulate clot. the list of path for samples from non-PE CTA
    "sample_dir_list": ["/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/non_PE/"
                        "initial_247/pe_v3_long_length_complete_vessel/denoise_low-resolution",
                        "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/non_PE/"
                        "initial_247/pe_v3_long_length_complete_vessel/original_low-resolution"],

    # do not simulate clot, direct use sample with human annotation
    # the list of path for samples with ground truth annotation
    "sample_dir_list_with_gt": ["/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "initial_50/pe_v3_long_length_complete_vessel/denoise_low-resolution",
                                "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "initial_50/pe_v3_long_length_complete_vessel/original_low-resolution",
                                "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "next_82/pe_v3_long_length_complete_vessel/original_low-resolution",
                                "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "next_82/pe_v3_long_length_complete_vessel/denoise_low-resolution",
                                "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "loop_3_59/pe_v3_long_length_complete_vessel/original_low-resolution",
                                "/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/"
                                "loop_3_59/pe_v3_long_length_complete_vessel/denoise_low-resolution"
                                ],

    "sample_interval": (0, 1),
    "shuffle_path_list": True,
    "checkpoint_dir":
        "/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_3/",

    "saved_model_filename": "model_guided.pth",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_file_name": None,
    "important_file_name": None,

    # -------------------------------------------------------
    # clot simulation specifics
    "top_dict_clot_pickle":
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/clot_pickle/complete',

    "clot_volume_range": (2000, 30000),  # the raw volume of the clot seed
    "num_clot_each_sample_range": (3, 3),  # number of clots applied to each sample sequence
    "min_clot": 300,  # the volume of clot when applied to vessel,
                      # 300-600 for high resolution, 50-300 for low resolution

    # simulation function 1: direct change the signal for clots
    "global_bias_range": [0, 0],  # set to None if you want to resume previous model's global_bias_range;
    # start from large minus like (-0.2, -1);
    # stable at (0, 0)

    # simulation function 2: add shift for clot regions
    "value_increase": None,  # [-0.025, -0.125],  # set to None if you want to resume previous model's value_increase;
    # very important parameter, larger the easier.
    # start with large value like (-0.5, -2.5),
    # for which the model can get very high performance, then gradually decrease it
    "voxel_variance": (0.8, 1),
    "power_range": (-0.3, 0.6),
    "add_base_range": (0, 3),

    # frequency for different clot simulation approach (will be normalize)
    "relative_frequency_v1_v2": [1, 0],  # (simulation_function 1, simulation_function 2)
    # frequency for different clot ground truth (will be normalize)
    "relative_frequency_simulate_gt": [2, 1],  # (simulated_clot, annotated_clot)

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
    parameters["sample_interval"] = (0, 10)

if __name__ == '__main__':
    train_iterations.training(parameters)
