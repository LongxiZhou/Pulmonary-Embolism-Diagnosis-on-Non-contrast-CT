"""
given a sequence (list of dict)
"""
import pulmonary_embolism_v3.utlis.phase_control_and_sample_process as utlis
import torch
import torch.nn as nn
import pulmonary_embolism_v3.models.model_transformer as model_transformer
import pulmonary_embolism_v3.prepare_training_dataset.trim_refine_and_remove_bad_scan as trim_sequence_length
import os


def load_saved_model_guided(model_path=None, high_resolution=False):
    if model_path is None:
        if high_resolution:
            model_path = \
                '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/high_resolution_warm_up/' \
                'gb_-0.00044_dice_0.836_recall_phase_model_guided.pth'
        else:
            # current best model
            model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/' \
                         'gb_0_dice_0.799_precision_phase_model_guided.pth'
            """
            # older version, train with 50 annotated samples
            model_path = \
                '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                'low_resolution_complete_vessel_long_v1_gt/best_model_guided.pth'
            """
    params = {
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
        "mlp_ratio": 2.0,

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    model = model_transformer.GuidedWithBranch(
        params["cube_size"], params["in_channel"], params["cnn_features"], params["given_features"],
        params["embed_dim"], params["num_heads"], params["encoding_depth"], params["interaction_depth"],
        params["decoding_depth"], params["segmentation_depth"], params["mlp_ratio"]
    )
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])

    data_dict = torch.load(model_path)
    if type(model) == nn.DataParallel:
        model.module.load_state_dict(data_dict["state_dict"])
    else:
        model.load_state_dict(data_dict["state_dict"])

    return model


def predict_clot_for_sample_sequence(list_sample_sequence, model=None, model_path=None, trim=False,
                                     high_resolution=False):
    """

    :param high_resolution:
    :param trim:
    :param list_sample_sequence: list of sample_sequence, or sample_sequence
    :param model: the model_guided on GPU, or None
    :param model_path:
    :return:
    add the key "clot_array" for each sample in list_sample_sequence
    add the key "certainty_array" for each sample in list_sample_sequence
    """
    assert len(list_sample_sequence) > 0

    strip = False  # whether input is list of sequence or just sequence

    if type(list_sample_sequence[0]) == dict:
        list_sample_sequence = [list_sample_sequence]
        strip = True

    if trim:
        print("trimming input sequence...")
        list_trimmed_sample_sequence = []
        for sample_sequence in list_sample_sequence:
            if high_resolution:
                trimmed_sample_sequence = trim_sequence_length.reduce_sequence_length(
                    sample_sequence, target_length=4000, max_branch=7)
            else:
                trimmed_sample_sequence = trim_sequence_length.reduce_sequence_length(
                    sample_sequence, target_length=3000, max_branch=9)
            list_trimmed_sample_sequence.append(trimmed_sample_sequence)
        list_sample_sequence = list_trimmed_sample_sequence

    params = {
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
        "mlp_ratio": 2.0,

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    if model is None:
        model = load_saved_model_guided(model_path, high_resolution=high_resolution)

    batch_size = len(list_sample_sequence)

    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region, cube_shape, clot_gt_tensor, \
            penalty_weight_tensor = \
            utlis.prepare_tensors_pe_transformer(list_sample_sequence, params["embed_dim"], device='cuda:0',
                                                 training_phase=False, roi='blood_vessel')

        segmentation_before_softmax = model(
            batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region)
        # [B, 2, N, flatten_dim]

        segment_probability_clot = softmax_layer(segmentation_before_softmax)[:, 1, :, :]
        segment_positive_certainty = segmentation_before_softmax[:, 1, :, :] - segmentation_before_softmax[:, 0, :, :]
        # [B, N, flatten_dim]
        # print(torch.min(segment_probability_clot), torch.max(segment_probability_clot))

        segment_probability_clot = \
            utlis.post_process_to_tensor(segment_probability_clot, (5, 5, 5))[:, :, 0, :, :, :].cpu().numpy()
        # [B, N, X, Y, Z]

        segment_positive_certainty = \
            utlis.post_process_to_tensor(segment_positive_certainty, (5, 5, 5))[:, :, 0, :, :, :].cpu().numpy()

        for i in range(batch_size):
            for j in range(len(list_sample_sequence[i])):
                item = list_sample_sequence[i][j]
                item['clot_prob_mask'] = segment_probability_clot[i, j, :, :, :]
                item['clot_certainty_mask'] = segment_positive_certainty[i, j, :, :, :]

    if strip:
        return list_sample_sequence[0]

    return list_sample_sequence


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    exit()
