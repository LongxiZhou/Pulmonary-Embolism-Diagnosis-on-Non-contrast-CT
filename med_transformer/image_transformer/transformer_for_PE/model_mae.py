import torch
import torch.nn as nn
from med_transformer.building_blocks import PatchEmbed3D, Mlp, GuidedBlock, flatten_batch
from med_transformer.utlis import init_weights_vit


class MAEGuidedSkipConnect(nn.Module):
    """
    The forward encoding is guided by the input information.
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
                batch_mask_info, batch_mask_query):
        """

        :param batch_tensor: in shape [batch_size, input_channel, X, Y, Z * input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :param query_vector: in shape [batch_size, num_query_cubes, positional embeddings]
        :param batch_mask_info, torch FloatTensor in shape [batch_size, input_channel, X, Y, Z * input_sequence_len]
        :param batch_mask_query,  torch FloatTensor in shape [batch_size, input_channel, X, Y, Z * query_sequence_len]
        :return:
        """

        # flatten_batch in shape [B, num_input_cubes, flatten_dim]
        flatten_info = flatten_batch(batch_tensor, self.cube_flatten_dim)

        flatten_mask_info = flatten_batch(batch_mask_info, self.cube_flatten_dim)
        flatten_mask_query = flatten_batch(batch_mask_query, self.cube_flatten_dim)

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, flatten_info, given_vector)

        information_vectors = self.forward_mid(vector_stack, flatten_info)

        prediction_vectors = self.forward_decoder(information_vectors, query_vector,
                                                  flatten_mask_info, flatten_mask_query)

        return prediction_vectors


if __name__ == '__main__':
    exit()
