"""
given a sequence (list of dict)
"""
import med_transformer.utlis as utlis
import torch
import torch.nn as nn
import med_transformer.image_transformer.transformer_for_PE.model_mae as model_mae
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def load_saved_model_guided(model_path=None, high_resolution=False):
    # load MaskedAutoEncoderModel
    if model_path is None:
        model_path = \
            '/home/zhoul0a/Desktop/pulmonary_embolism/check_point_guide/converge/predict_vessel/current_model.pth'
    params = {
        "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
        "in_channel": 1,  # 1 for CT
        "embed_dim": 384,  # the embedding dimension for each input cubes.
        # Require: embed_dim % int(6 * encoder_heads) == 0
        "given_dim": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
        "encoder_depth": 2,  # how many encoder encoding_blocks for encoding
        "encoder_heads": 16,  # for each encoder, how many attention heads
        "decoder_embed_dim": 384,  # the embedding dimension for decoding phase.
        # Require: decoder_embed_dim % decoder_num_heads == 0
        "decoder_depth": 1,  # how many decoder encoding_blocks
        "extra_decoder_depth": 1,
        "decoder_heads": 16,  # for each decoder, how many attention heads
        "mlp_ratio": 2.0,  # the DNN setting: len(vector) -> int(mlp_ratio * len(vector)) -> len(vector)

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
    if high_resolution:
        params["decoder_embed_dim"] = 192
        params["embed_dim"] = 192
    model = model_mae.MAEGuidedSkipConnect(params["cube_size"], params["in_channel"], params["embed_dim"],
                                           params["given_dim"], params["encoder_depth"],
                                           params["encoder_heads"],
                                           params["decoder_embed_dim"], params["decoder_depth"],
                                           params["decoder_heads"], params["mlp_ratio"], show=True,
                                           extra_decoder_depth=params["extra_decoder_depth"])
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


def predict_cube_sequence(list_information_sequence, list_query_depth_sequence, model=None, model_path=None,
                          high_resolution=False):
    """

    :param high_resolution:
    :param list_information_sequence: list of information_sequence, or information_sequence
    :param list_query_depth_sequence: list of query_depth_sequence, or query_depth_sequence, each is the dict contains:
    {'location_offset':, 'depth_cube':}
    :param model: the model_guided on GPU, or None
    :param model_path:
    :return: replace each location in list_query_sequence into dict
    {"ct_data": ct_cube, "location_offset": location_offset, "center_location": center_location}
    """
    assert len(list_information_sequence) > 0 and len(list_query_depth_sequence) > 0

    strip = False  # whether input is list of sequence or just sequence

    if type(list_information_sequence[0]) == dict:
        list_information_sequence = [list_information_sequence]
        list_query_depth_sequence = [list_query_depth_sequence]
        strip = True

    assert len(list_query_depth_sequence) == len(list_information_sequence)

    list_mass_center = []
    for information_sequence in list_information_sequence:
        dict_item = information_sequence[0]
        item_location_offset = dict_item["location_offset"]
        item_center_location = dict_item["center_location"]
        mass_center = (item_center_location[0] - item_location_offset[0],
                       item_center_location[1] - item_location_offset[1],
                       item_center_location[2] - item_location_offset[2])
        list_mass_center.append(mass_center)

    params = {
        "cube_size": (5, 5, 5),  # the shape of each cube, like (x, y, z)
        "in_channel": 1,  # 1 for CT
        "embed_dim": 384,  # the embedding dimension for each input cubes.
        # Require: embed_dim % int(6 * encoder_heads) == 0
        "given_dim": 0,  # the given dimensions for each input cubes, 0 for not use, len(given_vector) to use.
        "encoder_depth": 2,  # how many encoder encoding_blocks for encoding
        "encoder_heads": 16,  # for each encoder, how many attention heads
        "decoder_embed_dim": 384,  # the embedding dimension for decoding phase.
        # Require: decoder_embed_dim % decoder_num_heads == 0
        "decoder_depth": 1,  # how many decoder encoding_blocks
        "extra_decoder_depth": 1,
        "decoder_heads": 16,  # for each decoder, how many attention heads
        "mlp_ratio": 2.0,  # the DNN setting: len(vector) -> int(mlp_ratio * len(vector)) -> len(vector)

        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    if high_resolution:
        params["decoder_embed_dim"] = 192
        params["embed_dim"] = 192

    if model is None:

        model = load_saved_model_guided(model_path, high_resolution=high_resolution)

    list_query_sequence = []
    for query_gt_sequence in list_query_depth_sequence:
        query_loc_sequence = []
        for item in query_gt_sequence:
            query_loc_sequence.append(item['location_offset'])

        list_query_sequence.append(query_loc_sequence)

    batch_sample = {"list_sample_sequence": list_information_sequence,
                    "list_query_gt_sequence": list_query_depth_sequence}

    model.eval()
    with torch.no_grad():
        batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape = \
                utlis.prepare_tensors_3d_mae(list_information_sequence, list_query_sequence, params["embed_dim"],
                                             params["decoder_embed_dim"], params["given_dim"])

        batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query = \
            utlis.form_flatten_mask_mae(batch_sample)

        prediction_vectors = model(batch_tensor, pos_embed_tensor, given_vector, query_vector,
                                   batch_mask_info, batch_mask_query)

        list_predict_sequence = utlis.post_process_to_dict(prediction_vectors, list_query_sequence,
                                                           params["cube_size"], list_mass_center)

    if strip:
        return list_predict_sequence[0]

    return list_predict_sequence


if __name__ == '__main__':

    exit()
