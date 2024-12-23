import torch
import segment_clot_cta.training_pe_v3.training_iterations as train_iterations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parameters = {

    "mode": 'normal',  # ['temp', 'normal']  temp for debug, normal for training

    # --------------------------------------------------------
    # training specifics
    "difficulty": "stable",  # in ["increase", "decrease", "stable"]

    "resume": True,
    "reuse_phase_control": True,
    "reuse_optimizer": True,
    "reset_best_performance": False,
    "reset_outlier_detect": False,
    "augment": True,
    "trace_clot": True,
    "roi": "blood_vessel",  # region for search clot, in ["blood_vessel", "blood_region"],
    # but the clot simulation always in "blood_region"

    "n_epochs": 5000,

    "lr": 1e-3,
    "batch_size": 4,  # number of ct in each batch,    2 for lung region, 2 for all_file ct,
    "batch_size_test": 4,
    "accumulate_step": 32,  # real batch size is batch_ct * accumulate_step

    "num_prepared_dataset_test": 2,  # number of prepared dataset during testing, higher the more accuracy, but slower

    "target_recall": 0.6,
    "target_precision": None,
    "flip_recall": 0.6,  # reach this recall, then can flip
    "flip_precision": 0.6,  # reach this precision, then can flip
    "base_relative": 1.01,  # 1.01
    # will not flip util number times recall/precision bigger than precision/recall >= base_relative
    "base_recall": 0.,  # force flip when recall < base_recall
    "base_precision": 0.,  # force flip when precision < base_precision
    "max_performance_recall": 0.8,  # force flip when recall > max_performance in recall phase
    "max_performance_precision": 0.8,  # force flip when precision > max_performance in precision phase
    # flip when satisfy (flip_recall/precision and base_relative) or base_recall/precision or max_performance

    'min_dice_backup': 0.7,  # reach this dice, model may be saved as backup
    'min_dice': 0.05,  # reach this dice, model is considered as failed and will roll back to backup
    'min_dice_at_flip': 0.2,  # reach this dice at flip (recall == precision),
                              # model is considered as failed and will roll back to backup

    'min_dice_less_than_best': 0.3,  # if at flip, the dice is less than this amount to best model, then roll back

    "flip_remaining": 2,

    "final_phase": 'converge_to_recall',
    "warm_up_epochs": 2,
    "initial_relative_false_positive_penalty": 1.,

    # -------------------------------------------------------
    # dataloader specifics
    'num_workers': 32,  # num CPU for the parallel data loading, if you see load average < num_workers, reduce this
    "sample_dir_list": ["/data_disk/pulmonary_embolism/segment_clot_on_CTA/non_PE_CTA/sample_sequence/"
                        "pe_v3_long_length/"
                        "denoise_high-resolution",
                        "/data_disk/pulmonary_embolism/segment_clot_on_CTA/non_PE_CTA/sample_sequence/"
                        "pe_v3_long_length/"
                        "original_high-resolution"],

    "sample_interval": (0, 1),
    "shuffle_path_list": True,
    "checkpoint_dir":
        "/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/high_resolution_stable/",

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
    "num_clot_each_sample_range": (0, 3),  # number of clots applied to each sample sequence
    "min_clot": 300,  # the volume of clot when applied to vessel

    "global_bias_range": (0, 0),  # (-0.2, -1),

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
