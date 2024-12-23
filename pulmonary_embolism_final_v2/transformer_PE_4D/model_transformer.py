import torch
import torch.nn as nn
from med_transformer.building_blocks import PatchEmbed3D, Mlp, GuidedBlock, Block, flatten_batch
from med_transformer.utlis import init_weights_vit


class GuidedWithBranch(nn.Module):
    """
    Transformer flow has four phases:
    1) The encoding phase, is guided by the input information.
    2) The interaction phase, is formed by classic Transformer encoding_blocks.
    3) The decoding phase, is guided by the blood mask
    4) The segmentation phase, is guided by the blood mask

    final output is the clot possibility mask BEFORE Softmax.

    model input with shape:  [batch_size, num_cubes, image_channel * X * Y * Z]
    model output with shape: [batch_size, 2, num_cubes, image_channel * X * Y * Z], for the second channel, 0 for not
    clot, 1 for clot.

    Model inputs: see function "forward"
    batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * self.input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * self.input_sequence_len]
    pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
    given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
    query_vectors: in shape [batch_size, num_query_cubes, positional embeddings]
               the information_vectors and query_vectors are concatenated by dim=1
    """

    def __init__(self, cube_size=(5, 5, 5), in_channel=1, cnn_features=128, given_features=0,
                 embed_dim=128, num_heads=16, encoding_depth=2, interaction_depth=4, decoding_depth=2,
                 segmentation_depth=1, mlp_ratio=3., norm_layer=nn.LayerNorm, show=True):

        super().__init__()

        if show:
            print("cube_size:", cube_size, "\nin_channels:", in_channel, "\nembed_dim:", embed_dim,
                  "\ncnn_features, given_features:", cnn_features, given_features,
                  "\nnum_heads:", num_heads, "\nencoding_depth:", encoding_depth,
                  "\ninteraction_depth:", interaction_depth, "\ndecoding_depth:", decoding_depth,
                  "\nsegmentation_depth:", segmentation_depth,
                  "\nmlp_ratio:", mlp_ratio, "\nnorm_layer:", type(norm_layer))

        assert embed_dim % int(8 * num_heads) == 0
        assert encoding_depth >= 0 and interaction_depth >= 0 and decoding_depth >= 0
        assert encoding_depth + interaction_depth + decoding_depth > 0

        cube_flatten_dim = int(cube_size[0] * cube_size[1] * cube_size[2] * in_channel)
        self.cube_flatten_dim = cube_flatten_dim
        self.cube_size = cube_size
        self.norm_layer = norm_layer(embed_dim)
        self.embed_dim = embed_dim

        # --------------------------------------------------------------------------
        # encoding phase specifics

        # the layer for CNN encoding
        self.patch_embed = PatchEmbed3D(
            cube_size=cube_size, in_channels=in_channel, embed_dim=cnn_features, norm_layer=norm_layer)
        # input   [batch_size, in_channel, cube_size[0], cube_size[1], cube_size[2] * number_input_cubes]
        # output  [batch_size, number_input_cubes, cnn_features]

        # the layer combines CNN and flatten cube
        self.mlp_merged_flatten_input = Mlp(in_features=cnn_features + cube_flatten_dim, out_features=embed_dim,
                                            hidden_features=int(embed_dim * mlp_ratio))

        # the layer combines input data and the prior data (given features)
        self.initialize_feature_vector = Mlp(in_features=embed_dim + given_features, out_features=embed_dim,
                                             hidden_features=int(embed_dim * mlp_ratio))
        # the output of the layer is the features for each cube

        # the encoder is formed by GuidedBlock, guide vector is the output of self.initialize_feature_vector
        block_list_encoder = []
        for i in range(encoding_depth):
            block_list_encoder.append(
                GuidedBlock(embed_dim, num_heads, embed_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.encoding_blocks = nn.ModuleList(block_list_encoder)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # interaction phase specifics
        block_list_interaction = []
        for i in range(interaction_depth):
            block_list_interaction.append(
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.interaction_blocks = nn.ModuleList(block_list_interaction)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #  decoding phase specifics
        block_list_decoder = []
        for i in range(decoding_depth):
            block_list_decoder.append(
                GuidedBlock(embed_dim, num_heads, cube_flatten_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(block_list_decoder)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #  segmentation phase specifics
        block_list_form_positive = []
        for i in range(segmentation_depth):
            block_list_form_positive.append(
                GuidedBlock(embed_dim, num_heads, cube_flatten_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.segmentation_blocks_positive = nn.ModuleList(block_list_form_positive)

        block_list_form_negative = []
        for i in range(segmentation_depth):
            block_list_form_negative.append(
                GuidedBlock(embed_dim, num_heads, cube_flatten_dim, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.segmentation_blocks_negative = nn.ModuleList(block_list_form_negative)

        # cast to cube, cube_flatten_dim is X * Y * Z * out_channel
        self.pred_positive = nn.Linear(embed_dim, cube_flatten_dim, bias=True)
        self.pred_negative = nn.Linear(embed_dim, cube_flatten_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights_vit)

    def forward_encoder(self, batch_tensor, pos_embed_tensor, flatten_input, given_vector=None):
        """
        :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * num_input_cubes],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * num_input_cubes]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param flatten_input: in shape [batch_size, num_input_cubes, flatten_dim], flatten of the input cubes
        :param given_vector: in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :return: embedded vectors, in shape [batch_size, num_input_cubes, embedding dimension], like [1, 100, 1280]
        """
        # embed patches with CNN
        feature_cnn = self.patch_embed(batch_tensor)  # in shape [B, N, cnn_features]

        feature_vector = torch.cat((feature_cnn, flatten_input), dim=2)  # in shape [B, N, cnn_features + flatten_dim]
        feature_vector = self.mlp_merged_flatten_input(feature_vector)  # in shape [B, N, embed_dim]
        # the feature vector combines embed of CNN and DNN, will guide the encoding phase

        if given_vector is not None:
            feature_vector = torch.cat((feature_vector, given_vector), dim=2)  # in shape [B, N, embed_dim + given_dim]
            feature_vector = self.initialize_feature_vector(feature_vector)  # x in shape [B, N, embed_dim]
        else:
            feature_vector = self.initialize_feature_vector(feature_vector)  # x in shape [B, N, embed_dim]
        # now feature_vector combines CNN, DNN, and given vector

        # add pos embed, form the vector_stack to input the transformer blocks
        vector_stack = feature_vector + pos_embed_tensor

        # apply Transformer encoding_blocks
        for blk_guide in self.encoding_blocks:
            vector_stack = blk_guide(vector_stack, feature_vector)

        if len(self.encoding_blocks) > 0:
            vector_stack = self.norm_layer(vector_stack)

        return vector_stack  # in shape [B, N, embed_dim]

    def forward_interaction(self, vector_stack):
        """
        contains several classic transformer blocks

        :param vector_stack: the output of function "forward_encoder" in shape [B, N, embed_dim]

        :return: vector_stack
        """

        for blk in self.interaction_blocks:
            vector_stack = blk(vector_stack)
        if len(self.interaction_blocks) > 0:
            vector_stack = self.norm_layer(vector_stack)

        return vector_stack  # [B, N, embed_dim]

    def forward_decoder(self, vector_stack, flatten_vessel_mask):
        """
        the decoding is guided by the flatten_vessel_mask
        :param vector_stack: the output of function "forward_interaction" in shape [B, N, embed_dim]
        :param flatten_vessel_mask, torch FloatTensor in shape [B, N, flatten_dim]
        :return: vector_stack
        """

        # apply Transformer encoding_blocks
        for blk_guide in self.decoder_blocks:
            vector_stack = blk_guide(vector_stack, flatten_vessel_mask)
        if len(self.decoder_blocks) > 0:
            vector_stack = self.norm_layer(vector_stack)

        return vector_stack

    def forward_segmentation_negative(self, vector_stack, flatten_vessel_mask):
        """

        :param vector_stack: the output of function "forward_decoder" in shape [B, N, embed_dim]
        :param flatten_vessel_mask: torch FloatTensor in shape [B, N, flatten_dim]
        :return: [batch_size, num_query_cubes, prediction_len]
        here prediction_len = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """
        # apply Transformer encoding_blocks
        for blk_guide in self.segmentation_blocks_negative:
            vector_stack = blk_guide(vector_stack, flatten_vessel_mask)
        if len(self.segmentation_blocks_negative) > 0:
            vector_stack = self.norm_layer(vector_stack)

        prediction_vectors_negative = self.pred_negative(vector_stack)

        return prediction_vectors_negative

    def forward_segmentation_positive(self, vector_stack, flatten_vessel_mask):
        """

        :param vector_stack: the output of function "forward_decoder" in shape [B, N, embed_dim]
        :param flatten_vessel_mask: torch FloatTensor in shape [B, N, flatten_dim]
        :return: [batch_size, num_query_cubes, prediction_len]
        here prediction_len = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        """
        # apply Transformer encoding_blocks
        for blk_guide in self.segmentation_blocks_positive:
            vector_stack = blk_guide(vector_stack, flatten_vessel_mask)
        if len(self.segmentation_blocks_positive) > 0:
            vector_stack = self.norm_layer(vector_stack)

        prediction_vectors_positive = self.pred_positive(vector_stack)

        return prediction_vectors_positive

    def forward(self, batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask):
        """

        :param batch_tensor: in shape [batch_size, input_channel, X, Y, Z * input_sequence_len],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * input_sequence_len]
        :param pos_embed_tensor: in shape [batch_size, num_input_cubes, embedding dimension], in torch.FloatTensor
        :param given_vector: None or in shape [batch_size, num_input_cubes, num_given_values], in torch.FloatTensor
        :param flatten_vessel_mask, torch FloatTensor in shape [B, N, flatten_dim]
        :return: probability mask for clot, not softmax.
        [batch_size, 2, num_query_cubes, prediction_len]
        here prediction len = image_channel * X * Y * Z, resize prediction vector to get the predicted CT value
        for second channel, 0 for negative, 1 for clot
        """

        # print("tracking...")

        # flatten_input in shape [B, num_input_cubes, flatten_dim]
        flatten_input = flatten_batch(batch_tensor, self.cube_flatten_dim)
        # print(flatten_input.size())

        vector_stack = self.forward_encoder(batch_tensor, pos_embed_tensor, flatten_input, given_vector)
        # print(vector_stack.size())

        vector_stack = self.forward_interaction(vector_stack)
        # print(vector_stack.size())

        vector_stack = self.forward_decoder(vector_stack, flatten_vessel_mask)
        # print(vector_stack.size())

        prediction_vector_negative = self.forward_segmentation_negative(vector_stack, flatten_vessel_mask)
        prediction_vector_positive = self.forward_segmentation_positive(vector_stack, flatten_vessel_mask)

        segmentation_before_softmax = torch.stack((prediction_vector_negative, prediction_vector_positive), dim=1)

        return segmentation_before_softmax  # [B, 2, N, dim_flatten]


if __name__ == '__main__':
    exit()
