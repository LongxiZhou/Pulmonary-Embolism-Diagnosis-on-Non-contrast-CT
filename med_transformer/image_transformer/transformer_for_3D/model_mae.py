import torch
import torch.nn as nn
import numpy as np
from med_transformer.building_blocks import PatchEmbed3D, Block, flatten_batch, Mlp, GuidedBlock
from med_transformer.position_embeding import get_3d_sincos_pos_embed_loc_list
from med_transformer.utlis import init_weights_vit, post_process_to_list, post_process_to_tensor


class MaskedAutoEncoderList(nn.Module):
    """
    Model inputs: see function "forward"
    1) list of information sequence: a list of lists of dict, each dict contains {'ct_data': ct_cube,
    'location_offset': center_location_offset, 'given_vector': given_vector}

    list of information sequence = [[dict, dict, dict, ...], [dict, dict, dict, ...], ...], the length is batch size
    data on CPU

    2) list of query sequence: a list of lists of location offsets, like (-10, 12, 13)

    list of query sequence = [[tuple, tuple, tuple, ...], [tuple, tuple, tuple, ...], ...], the length is batch size
    data on CPU

    Model outputs: a torch.FloatTensor
    for each tuple in the Query sequence, return the predicted value for "ct_data", e.g., ct_cubes in (5, 5, 5)

    outputs = [[array, array, array, ...], [array, array, array, ...], ...]
    outputs.shape = [batch_size, max(len(query_sequence in the batch))), cube_shape_x, cube_shape_y, cube_shape_z]
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "decoder_num_heads:", decoder_num_heads, "mlp_ratio:", mlp_ratio,
                  "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int(embed_dim * mlp_ratio))

        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        block_list_decoder = []
        for i in range(decoder_depth):
            block_list_decoder.append(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                                            norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.num_output_voxel = cube_size[0] * cube_size[1] * cube_size[2] * in_chan

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_output_voxel, bias=True)  # decoder to cube
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]
        if given_vector is not None:
            x = torch.cat((x, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
        x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack):
        """

        :param vector_stack: the output of function "forward_encoder"
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """
        information_vectors = self.decoder_embed(vector_stack)
        return information_vectors

    def forward_decoder(self, information_vectors, query_vectors):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
        :return: [batch_size, num_query_cubes, prediction_vector],
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks
        for blk in self.decoder_blocks:
            vector_pack = blk(vector_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        # remove information token
        vector_pack = vector_pack[:, information_vectors.shape[1]:, :]

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def prepare_tensor_inputs(self, list_information_sequence, list_query_sequence):
        # prepare batch_tensor, pos_embed_tensor, given_vector, query_vector,
        batch_size = len(list_information_sequence)
        assert batch_size == len(list_query_sequence) and batch_size > 0

        for i in range(batch_size):
            if len(list_information_sequence[i]) > self.input_sequence_len:
                self.input_sequence_len = len(list_information_sequence[i])

        for i in range(batch_size):
            if len(list_query_sequence[i]) > self.query_sequence_len:
                self.query_sequence_len = len(list_query_sequence[i])

        cube_shape = np.shape(list_information_sequence[0][0]['ct_data'])
        batch_array = np.zeros([batch_size, 1, cube_shape[0], cube_shape[1],
                                cube_shape[2] * self.input_sequence_len], 'float32')
        location_list = []

        given_vector_array = np.zeros([batch_size, self.input_sequence_len, self.given_dim], 'float32')

        # complete batch_array, source_arrays and given_vector_array
        for i in range(batch_size):
            for j in range(len(list_information_sequence[i])):
                item = list_information_sequence[i][j]
                batch_array[i, 0, :, :, j * cube_shape[2]: (j + 1) * cube_shape[2]] = item['ct_data']
                location_list.append(item['location_offset'])
                if self.given_dim > 0:
                    given_vector_array[i, j, :] = item['given_vector']

        pos_embed_array = get_3d_sincos_pos_embed_loc_list(self.embed_dim, location_list)
        pos_embed_array_final = np.zeros([batch_size, self.input_sequence_len, self.embed_dim], 'float32')
        shift = 0
        for i in range(batch_size):
            for j in range(len(list_information_sequence[i])):
                pos_embed_array_final[i, j, :] = pos_embed_array[shift, :]
                shift += 1

        # prepare input for function "forward_encoder"
        batch_tensor = torch.FloatTensor(batch_array).cuda()
        pos_embed_tensor = torch.FloatTensor(pos_embed_array_final).cuda()
        if self.given_dim > 0:
            given_vector = torch.FloatTensor(given_vector_array).cuda()
        else:
            given_vector = None

        # prepare query_vectors, which is in [batch_size, self.query_sequence_len, self.decoding_dim]
        location_list_query = []
        for i in range(batch_size):
            for j in range(self.query_sequence_len):
                if j < len(list_query_sequence[i]):
                    location_list_query.append(list_query_sequence[i][j])
                else:
                    location_list_query.append((500, 500, 500))  # appended an outlier
        pos_embed_array_query = get_3d_sincos_pos_embed_loc_list(self.decoding_dim, location_list_query)
        pos_embed_array_query = np.reshape(pos_embed_array_query,
                                           [batch_size, self.query_sequence_len, self.decoding_dim])
        query_vector = torch.FloatTensor(pos_embed_array_query).cuda()

        return batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape

    def forward(self, list_information_sequence, list_query_sequence, return_tensor=True):

        batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape = self.prepare_tensor_inputs(
            list_information_sequence, list_query_sequence)

        #################################################################################
        # feed into the model_guided
        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, given_vector)

        information_vectors = self.forward_mid(vector_stack)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector)

        if return_tensor:
            self.input_sequence_len = 0
            self.query_sequence_len = 0
            return post_process_to_tensor(prediction_vectors, cube_shape)
            # in shape [batch_size, num_query_cubes, 1, X, Y, Z]

        else:  # this is for inference
            self.input_sequence_len = 0
            self.query_sequence_len = 0
            predicted_ct_cubes = post_process_to_list(prediction_vectors, list_query_sequence, cube_shape)
            return predicted_ct_cubes  # replace the locations in list_query_sequence into ct_cubes in numpy, on CPU.


class MaskedAutoEncoderTensorV1(nn.Module):
    """
    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "decoder_num_heads:", decoder_num_heads, "mlp_ratio:", mlp_ratio,
                  "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int(embed_dim * mlp_ratio))

        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        block_list_decoder = []
        for i in range(decoder_depth):
            block_list_decoder.append(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                                            norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.num_output_voxel = cube_size[0] * cube_size[1] * cube_size[2] * in_chan

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_output_voxel, bias=True)  # decoder to cube
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]
        if given_vector is not None:
            x = torch.cat((x, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
        x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack):
        """

        :param vector_stack: the output of function "forward_encoder"
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """
        information_vectors = self.decoder_embed(vector_stack)
        return information_vectors

    def forward_decoder(self, information_vectors, query_vectors):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
        :return: [batch_size, num_query_cubes, prediction_vector],
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks
        for blk in self.decoder_blocks:
            vector_pack = blk(vector_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        vector_pack = vector_pack[:, information_vectors.shape[1]:, :]

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector):

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, given_vector)

        information_vectors = self.forward_mid(vector_stack)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector)

        return prediction_vectors


class MaskedAutoEncoderTensorV2(nn.Module):
    """
    The difference of V1 and V2 is on the forward section.
    class MaskedAutoEncoderTensorV2 has a attribute "train",
    if self.train is True, the output has shape:
    [batch_size, num_information_cubes + num_query_cubes, image_channel * X * Y * Z)]
    if self.train is False, the output has shape:
    [batch_size, num_query_cubes, image_channel * X * Y * Z)]

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "decoder_num_heads:", decoder_num_heads, "mlp_ratio:", mlp_ratio,
                  "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int(embed_dim * mlp_ratio))

        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        block_list_decoder = []
        for i in range(decoder_depth):
            block_list_decoder.append(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                                            norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.num_output_voxel = cube_size[0] * cube_size[1] * cube_size[2] * in_chan

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_output_voxel, bias=True)  # decoder to cube
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]
        if given_vector is not None:
            x = torch.cat((x, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
        x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack):
        """

        :param vector_stack: the output of function "forward_encoder"
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """
        information_vectors = self.decoder_embed(vector_stack)
        return information_vectors

    def forward_decoder(self, information_vectors, query_vectors):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
        :return:
        [batch_size, num_query_cubes, prediction_vector] during inference
        or [batch_size, num_information_cubes num_query_cubes, prediction_vector] during training
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks
        for blk in self.decoder_blocks:
            vector_pack = blk(vector_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        if not self.training:
            # remove information token during inference
            vector_pack = vector_pack[:, information_vectors.shape[1]:, :]

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector):

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, given_vector)

        information_vectors = self.forward_mid(vector_stack)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector)

        return prediction_vectors


class MaskedAutoEncoderTensorV3(nn.Module):
    """
    The difference of V2 and V3 is on the model_guided structure. For V2 the prediction for information cubes is the same
    architecture for predicting the query sequence, while for V3 the prediction for information cubes is decode by less
    decoder encoding_blocks: like 6 decoder layers to reconstruct the information cubes and 2 more decoder layers to reconstruct
    the query cubes. The decoder layer for information cubes share weights with query cubes.

    class MaskedAutoEncoderTensorV3 has a attribute "train",
    if self.train is True, the output has shape:
    [batch_size, num_information_cubes + num_query_cubes, image_channel * X * Y * Z)]
    if self.train is False, the output has shape:
    [batch_size, num_query_cubes, image_channel * X * Y * Z)]

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False, extra_decoder_depth=2):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "extra_decoder_depth:", extra_decoder_depth, "decoder_num_heads:",
                  decoder_num_heads, "mlp_ratio:", mlp_ratio, "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int(embed_dim * mlp_ratio))

        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        self.decoder_depth = decoder_depth
        self.extra_decoder_depth = extra_decoder_depth

        block_list_decoder = []
        for i in range(decoder_depth + extra_decoder_depth):
            block_list_decoder.append(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                                            norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.num_output_voxel = cube_size[0] * cube_size[1] * cube_size[2] * in_chan

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_output_voxel, bias=True)  # decoder to cube
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]
        if given_vector is not None:
            x = torch.cat((x, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
        x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack):
        """

        :param vector_stack: the output of function "forward_encoder"
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """
        information_vectors = self.decoder_embed(vector_stack)
        return information_vectors

    def forward_decoder(self, information_vectors, query_vectors):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
        :return:
        [batch_size, num_query_cubes, prediction_vector] during inference
        or [batch_size, num_information_cubes num_query_cubes, prediction_vector] during training
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks to decode information sequence
        for block_id in range(self.decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack)

        vector_pack_information = vector_pack.clone()[:, 0: information_vectors.shape[1], :]

        # apply extra Transformer encoding_blocks to decode query sequence
        for block_id in range(self.decoder_depth, self.decoder_depth + self.extra_decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        vector_pack_query = vector_pack[:, information_vectors.shape[1]:, :]

        if not self.training:
            # remove information token during inference
            vector_pack = vector_pack_query
        else:
            vector_pack = torch.cat((vector_pack_information, vector_pack_query), 1)

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector):

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, given_vector)

        information_vectors = self.forward_mid(vector_stack)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector)

        return prediction_vectors


class MAEForBloodVessel(nn.Module):
    """
    The class is based on the "MaskedAutoEncoderTensorV3", but the decoder is different.
    The decoder is specially designed for the blood vessel prediction:
    "self.forward_mid" flatten the vessel mask and concatenate to the embedding vector after cast to vector for
    decoding, so the vectors feed to decoder is of dim decoding_dim + flatten_dim;
    "self.forward_decoder" also flatten the vessel mask and concatenate to the vector before the final projection.

    class MAEForBloodVessel has a attribute "train",
    if self.train is True, the output has shape:
    [batch_size, num_information_cubes + num_query_cubes, image_channel * X * Y * Z)]
    if self.train is False, the output has shape:
    [batch_size, num_query_cubes, image_channel * X * Y * Z)]

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False, extra_decoder_depth=2):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "extra_decoder_depth:", extra_decoder_depth, "decoder_num_heads:",
                  decoder_num_heads, "mlp_ratio:", mlp_ratio, "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int(embed_dim * mlp_ratio))

        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0

        cube_flatten_dim = int(cube_size[0] * cube_size[1] * cube_size[2] * in_chan)
        self.cube_flatten_dim = cube_flatten_dim

        # flatten the original cube and skip feed to the decoder
        self.decoder_embed = nn.Linear(embed_dim + self.cube_flatten_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        self.decoder_depth = decoder_depth
        self.extra_decoder_depth = extra_decoder_depth

        block_list_decoder = []
        # the decoder is formed by GuidedBlock, guide vector is the flatten of the vessel mask
        for i in range(decoder_depth + extra_decoder_depth):
            block_list_decoder.append(GuidedBlock(self.decoding_dim, decoder_num_heads, cube_flatten_dim,
                                                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer))

        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(self.decoding_dim)

        # decoder to cube, cube_flatten_dim is the flatten of the vessel mask
        self.decoder_pred = nn.Linear(self.decoding_dim + cube_flatten_dim, self.cube_flatten_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]

        if given_vector is not None:
            x = torch.cat((x, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
        x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack, flatten_info):
        """

        :param vector_stack: the output of function "forward_encoder" in shape [B, N, embed_dim]
        :param flatten_info: the flatten of batch_tensor [batch, num_input_cubes, flatten_dim]
        in torch.FloatTensor, like [2, 100, 125]
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """

        information_vectors = torch.cat((vector_stack, flatten_info), 2)  # [B, N, embed_dim + flatten_dim]

        information_vectors = self.decoder_embed(information_vectors)

        return information_vectors  # [B, N, decoding_dim]

    def forward_decoder(self, information_vectors, query_vectors, flatten_mask_info, flatten_mask_query):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1

        :param flatten_mask_info: in shape [batch_size, num_information_cubes, flatten_dim]
        :param flatten_mask_query: in shape [batch_size, num_query_cubes, flatten_dim]

        :return:
        [batch_size, num_query_cubes, prediction_vector] during inference
        or [batch_size, num_information_cubes num_query_cubes, prediction_vector] during training
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """
        # concatenate flatten_mask_info and flatten_mask_query
        flatten_mask = torch.cat((flatten_mask_info, flatten_mask_query), 1)  # [B, n_i + n_q, D_flatten]

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks to decode information sequence
        for block_id in range(self.decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack, flatten_mask)

        vector_pack_information = vector_pack.clone()[:, 0: information_vectors.shape[1], :]

        # apply extra Transformer encoding_blocks to decode query sequence
        for block_id in range(self.decoder_depth, self.decoder_depth + self.extra_decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack, flatten_mask)

        # vector_pack = self.decoder_norm(vector_pack)

        vector_pack_query = vector_pack[:, information_vectors.shape[1]:, :]

        if not self.training:
            # remove information token during inference
            vector_pack = vector_pack_query  # [B, n_q, D]
            vector_pack = torch.cat((vector_pack, flatten_mask_query), 2)  # [B, n_q, D + D_flatten]
        else:
            vector_pack = torch.cat((vector_pack_information, vector_pack_query), 1)  # [B, n_i + n_q, D]
            vector_pack = torch.cat((vector_pack, flatten_mask), 2)  # [B, n_i + n_q, D + D_flatten]

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack  # [B, N, D_flatten]

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector, flatten_mask_info,
                flatten_mask_query):
        """

        :param batch_tensor: in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :param query_vector: in shape [batch_size, num_query_cubes, positional embeddings]
        :param flatten_mask_info: in shape [batch_size, num_input_cubes, flatten_dim], torch.FloatTensor
        :param flatten_mask_query: in shape [batch_size, num_query_cubes, flatten_dim], torch.FloatTensor
        :return:
        """

        # flatten_batch in shape [B, num_input_cubes, flatten_dim]
        flatten_info = flatten_batch(batch_tensor, self.cube_flatten_dim)

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, given_vector)

        information_vectors = self.forward_mid(vector_stack, flatten_info)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector, flatten_mask_info,
                                                  flatten_mask_query)

        return prediction_vectors


class MAESkipConnect(nn.Module):
    """
    The class is based on the "MaskedAutoEncoderTensorV3", but the embedding and forward_mid are different.

    embedding combines the flatten cube and the CNN features
    the

    class MAESkipConnect has a attribute "train",
    if self.train is True, the output has shape:
    [batch_size, num_information_cubes + num_query_cubes, image_channel * X * Y * Z)]
    if self.train is False, the output has shape:
    [batch_size, num_query_cubes, image_channel * X * Y * Z)]

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False, extra_decoder_depth=2):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "extra_decoder_depth:", extra_decoder_depth, "decoder_num_heads:",
                  decoder_num_heads, "mlp_ratio:", mlp_ratio, "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0

        cube_flatten_dim = int(cube_size[0] * cube_size[1] * cube_size[2] * in_chan)
        self.cube_flatten_dim = cube_flatten_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_extract_feature_vector = Mlp(in_features=embed_dim + cube_flatten_dim, out_features=embed_dim,
                                              hidden_features=int((embed_dim + cube_flatten_dim) * mlp_ratio))

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int((embed_dim + given_dim) * mlp_ratio))

        # the encoder is formed by GuidedBlock, guide vector is the flatten of the input feature
        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(
                GuidedBlock(embed_dim, num_heads, embed_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0

        #  skip feed to the decoder the input feature
        self.decoder_embed = nn.Linear(embed_dim + cube_flatten_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        self.decoder_depth = decoder_depth
        self.extra_decoder_depth = extra_decoder_depth

        block_list_decoder = []

        for i in range(decoder_depth + extra_decoder_depth):
            block_list_decoder.append(
                Block(self.decoding_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))

        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(self.decoding_dim)

        # decoder to cube, cube_flatten_dim is the flatten of the vessel mask
        self.decoder_pred = nn.Linear(self.decoding_dim, self.cube_flatten_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, flatten_info, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param flatten_info: in shape [batch_size, num_input_cubes, flatten_dim]
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]

        x = torch.cat((x, flatten_info), dim=2)  # x in shape [B, N, embed_dim + flatten_dim]

        feature_vector = self.mlp_extract_feature_vector(x)  # in shape [B, N, embed_dim]

        if given_vector is not None:
            x = torch.cat((feature_vector, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
            x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]
        else:
            x = self.mlp_embed(feature_vector)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x, feature_vector)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack, flatten_info):
        """

        :param vector_stack: the output of function "forward_encoder" in shape [B, N, embed_dim]
        :param flatten_info: the flatten of batch_tensor [batch, num_input_cubes, flatten_dim]
        in torch.FloatTensor, like [2, 100, 125]
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """

        information_vectors = torch.cat((vector_stack, flatten_info), 2)
        # [B, N, embed_dim + flatten_dim]

        information_vectors = self.decoder_embed(information_vectors)

        return information_vectors  # [B, N, decoding_dim]

    def forward_decoder(self, information_vectors, query_vectors):
        """

        :param information_vectors: output from function "forward_mid"
        :param query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
        :return:
        [batch_size, num_query_cubes, prediction_vector] during inference
        or [batch_size, num_information_cubes num_query_cubes, prediction_vector] during training
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        # apply Transformer encoding_blocks to decode information sequence
        for block_id in range(self.decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack)

        vector_pack_information = vector_pack.clone()[:, 0: information_vectors.shape[1], :]

        # apply extra Transformer encoding_blocks to decode query sequence
        for block_id in range(self.decoder_depth, self.decoder_depth + self.extra_decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        vector_pack_query = vector_pack[:, information_vectors.shape[1]:, :]

        if not self.training:
            # remove information token during inference
            vector_pack = vector_pack_query
        else:
            vector_pack = torch.cat((vector_pack_information, vector_pack_query), 1)

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector):
        """

        :param batch_tensor: in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :param query_vector: in shape [batch_size, num_query_cubes, positional embeddings]
        :return:
        """

        # flatten_batch in shape [B, num_input_cubes, flatten_dim]
        flatten_info = flatten_batch(batch_tensor, self.cube_flatten_dim)

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, flatten_info, given_vector)

        information_vectors = self.forward_mid(vector_stack, flatten_info)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector)

        return prediction_vectors


class MAEGuidedSkipConnect(nn.Module):
    """
    The class is based on the "MAESkipConnect", but the "forward_decoding" is different,
    the forward decoding is guided by the vessel mask.


    class MAEGuidedSkipConnect has a attribute "train",
    if self.train is True, the output has shape:
    [batch_size, num_information_cubes + num_query_cubes, image_channel * X * Y * Z)]
    if self.train is False, the output has shape:
    [batch_size, num_query_cubes, image_channel * X * Y * Z)]

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_chan=1,
                 embed_dim=1152, given_dim=0, depth=24, num_heads=16,
                 decoder_embed_dim=1152, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, show=False, extra_decoder_depth=2):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "in_channels:", in_chan, "embed_dim:", embed_dim, "given_dim:", given_dim,
                  "encoder_depth:", depth, "encoder_num_heads:", num_heads, "decoder_embed_dim:", decoder_embed_dim,
                  "decoder_depth:", decoder_depth, "extra_decoder_depth:", extra_decoder_depth, "decoder_num_heads:",
                  decoder_num_heads, "mlp_ratio:", mlp_ratio, "norm_layer:", type(norm_layer))

        assert embed_dim % int(6 * num_heads) == 0 and decoder_embed_dim % decoder_num_heads == 0

        cube_flatten_dim = int(cube_size[0] * cube_size[1] * cube_size[2] * in_chan)
        self.cube_flatten_dim = cube_flatten_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cube_size = cube_size
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_chan, embed_dim=embed_dim, norm_layer=norm_layer)

        self.mlp_extract_feature_vector = Mlp(in_features=embed_dim + cube_flatten_dim, out_features=embed_dim,
                                              hidden_features=int((embed_dim + cube_flatten_dim) * mlp_ratio))

        self.mlp_embed = Mlp(in_features=embed_dim + given_dim, out_features=embed_dim,
                             hidden_features=int((embed_dim + given_dim) * mlp_ratio))

        # the encoder is formed by GuidedBlock, guide vector is the flatten of the input feature
        block_list_encoder = []
        for i in range(depth):
            block_list_encoder.append(
                GuidedBlock(embed_dim, num_heads, embed_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(block_list_encoder)

        self.norm = norm_layer(embed_dim)
        self.input_sequence_len = 0
        self.batch_size = None
        self.given_dim = given_dim
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.query_sequence_len = 0

        #  skip feed to the decoder the input feature
        self.decoder_embed = nn.Linear(embed_dim + cube_flatten_dim, decoder_embed_dim, bias=True)

        self.decoding_dim = decoder_embed_dim

        self.decoder_depth = decoder_depth
        self.extra_decoder_depth = extra_decoder_depth

        block_list_decoder = []

        for i in range(decoder_depth + extra_decoder_depth):
            block_list_decoder.append(
                GuidedBlock(self.decoding_dim, decoder_num_heads, cube_flatten_dim, mlp_ratio,
                            qkv_bias=True, norm_layer=norm_layer))

        self.decoder_blocks = nn.ModuleList(block_list_decoder)

        self.decoder_norm = norm_layer(self.decoding_dim)

        # decoder to cube, cube_flatten_dim is the flatten of the vessel mask
        self.decoder_pred = nn.Linear(self.decoding_dim, self.cube_flatten_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, flatten_info, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param flatten_info: in shape [batch_size, num_input_cubes, flatten_dim]
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches
        x = self.patch_embed(batch_tensor)  # x in shape [B, N, embed_dim]

        x = torch.cat((x, flatten_info), dim=2)  # x in shape [B, N, embed_dim + flatten_dim]

        feature_vector = self.mlp_extract_feature_vector(x)  # in shape [B, N, embed_dim]

        if given_vector is not None:
            x = torch.cat((feature_vector, given_vector), dim=2)  # x in shape [B, N, embed_dim + given_dim]
            x = self.mlp_embed(x)  # x in shape [B, N, embed_dim]
        else:
            x = self.mlp_embed(feature_vector)  # x in shape [B, N, embed_dim]

        # add pos embed
        x = x + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk in self.blocks:
            x = blk(x, feature_vector)
        vector_stack = self.norm(x)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_mid(self, vector_stack, flatten_info):
        """

        :param vector_stack: the output of function "forward_encoder" in shape [B, N, embed_dim]
        :param flatten_info: the flatten of batch_tensor [batch, num_input_cubes, flatten_dim]
        in torch.FloatTensor, like [2, 100, 125]
        :return: embedded vectors from model_guided input, containing information for decoding
                 in shape [batch_size, num_input_cubes, embedding dimension for decoding]
        """

        information_vectors = torch.cat((vector_stack, flatten_info), 2)  # [B, N, embed_dim + flatten_dim]

        information_vectors = self.decoder_embed(information_vectors)

        return information_vectors  # [B, N, decoding_dim]

    def forward_decoder(self, information_vectors, query_vectors, flatten_mask_info, flatten_mask_query):
        """

        :param information_vectors: output from function "forward_mid"
        in shape [batch_size, num_input_cubes, dim_decoding]
        :param query_vectors: in shape [batch_size, num_query_cubes, positional_embeddings in dim_decoding]
               the information_vectors and query_vectors are concatenated by dim=1
        :param flatten_mask_info, torch FloatTensor in shape [batch_size, num_input_cubes, flatten_dim]
        :param flatten_mask_query,  torch FloatTensor in shape [batch_size, num_query_cubes, flatten_dim]
        :return:
        [batch_size, num_query_cubes, prediction_vector] during inference
        or [batch_size, num_information_cubes num_query_cubes, prediction_vector] during training
        here prediction vector = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """

        # concatenate information_vectors and query_vectors
        vector_pack = torch.cat((information_vectors, query_vectors), 1)  # [B, n_i + n_q, D]

        mask_pack = torch.cat((flatten_mask_info, flatten_mask_query), 1)  # [B, n_i + n_q, D_flatten]

        # apply Transformer encoding_blocks to decode information sequence
        for block_id in range(self.decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack, mask_pack)

        vector_pack_information = vector_pack.clone()[:, 0: information_vectors.shape[1], :]

        # apply extra Transformer encoding_blocks to decode query sequence
        for block_id in range(self.decoder_depth, self.decoder_depth + self.extra_decoder_depth):
            vector_pack = self.decoder_blocks[block_id](vector_pack, mask_pack)

        # vector_pack = self.decoder_norm(vector_pack)

        vector_pack_query = vector_pack[:, information_vectors.shape[1]:, :]

        if not self.training:
            # remove information token during inference
            vector_pack = vector_pack_query
        else:
            vector_pack = torch.cat((vector_pack_information, vector_pack_query), 1)

        # predictor projection
        vector_pack = self.decoder_pred(vector_pack)

        return vector_pack

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, query_vector,
                flatten_mask_info, flatten_mask_query):
        """

        :param batch_tensor: in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :param query_vector: in shape [batch_size, num_query_cubes, positional embeddings]
        :param flatten_mask_info, torch FloatTensor in shape [batch_size, num_input_cubes, flatten_dim]
        :param flatten_mask_query,  torch FloatTensor in shape [batch_size, num_query_cubes, flatten_dim]
        :return:
        """

        # flatten_batch in shape [B, num_input_cubes, flatten_dim]
        flatten_info = flatten_batch(batch_tensor, self.cube_flatten_dim)

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, flatten_info, given_vector)

        information_vectors = self.forward_mid(vector_stack, flatten_info)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector,
                                                  flatten_mask_info, flatten_mask_query)

        return prediction_vectors


if __name__ == '__main__':
    model = MaskedAutoEncoderList()

    model = model.to('cuda:0')
    import Tool_Functions.Functions as Functions

    temp_information_dict = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/temp/cube_dict_list.pickle')
    temp_information_dict = [temp_information_dict]
    temp_query_sequence = [[(1, 2, 3), (0.5, -4, -5), (6, 7, 9), (1, 1, 1), (1, 2, 3)]]

    array_list = model(temp_information_dict, temp_query_sequence, return_tensor=False)
    print(array_list)
    print(len(array_list), np.shape(array_list[0]))
    print(np.sum(np.abs(array_list[0][0] - array_list[0][4])))
    print(np.sum(np.abs(array_list[0][0] - array_list[0][3])))
    exit()
