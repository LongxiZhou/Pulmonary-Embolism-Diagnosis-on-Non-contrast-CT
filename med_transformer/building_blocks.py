from torch import nn as nn
import torch


class PatchEmbed2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chan=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.projection = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.projection(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C, N is the number of patches, C is the feature length
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Cube to Patch Embedding
    """
    def __init__(self, cube_size=(5, 5, 5), in_channels=1, embed_dim=1280, norm_layer=None):
        super().__init__()
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=cube_size, stride=cube_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """

        :param x: tensor in shape [batch_size, channel, cube_size[0], cube_size[1], cube_size[2] * number_input_cubes)]
        :return: [batch_size, number_input_cubes, embedding_dim]
        """
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x  # B, N, C


def flatten_batch(batch_tensor, flatten_dim):
    """
    :param batch_tensor: should in shape [batch_size, input_channel, X, Y, Z * num_input_cubes],
        in torch.FloatTensor, like [2, 1, 5, 5, 5 * num_input_cubes]
    :param flatten_dim:
    :return: tensor in [batch_size, number_input_cubes, flatten_dim], like [2, self.input_sequence_len, 125]
    """
    batch_size, channel, x, y, z_by_cube_count = batch_tensor.size()
    num_input_cubes = int(channel * x * y * z_by_cube_count / flatten_dim)
    z = int(z_by_cube_count / num_input_cubes)

    flatten = torch.swapaxes(batch_tensor, 1, 4)  # [batch_size, Z * num_input_cubes, X, Y, input_channel]
    flatten = torch.reshape(flatten, (batch_size, num_input_cubes, z, x, y, channel))
    return torch.reshape(flatten, (batch_size, num_input_cubes, flatten_dim))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GuidedMlp(nn.Module):
    """
    the input is concatenate with the guided_vector
    """
    def __init__(self, in_features, guide_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features + guide_features

        self.fc1 = nn.Linear(in_features + guide_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, input_vector, guide_vector):
        x = torch.cat((input_vector, guide_vector), dim=-1)
        # note linear only apply on the last dim [1, 2, 3, 4, 10] undergo nn.Linear(10, 20) become [1, 2, 3, 4, 20]

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic encoding_depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """

        :param x: [B, N, C]  B: batch_size; N: number patches; C: embedding dimension
        :return: in shape [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GuidedBlock(nn.Module):
    """
    the only difference of the class "GuidedBlock" with "Block" is that it use the "GuidedMlp"
    """
    def __init__(self, dim, num_heads, guide_features, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic encoding_depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.guided_mlp = GuidedMlp(in_features=dim, guide_features=guide_features,
                                    hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, guide_vector):
        """

        :param x: [B, N, C]  B: batch_size; N: number patches; C: embedding dimension
        :param guide_vector: [B, N, C_guid]  B: batch_size; N: number patches; C_guide: guide dimension

        :return: in shape [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.guided_mlp(self.norm2(x), guide_vector)
        return x


if __name__ == '__main__':
    test_array = torch.zeros((5, 5, 5))
    test_array_1 = test_array.clone().detach() + 1
    test_array_2 = test_array.clone().detach() + 2
    test_array_3 = test_array.clone().detach() + 3
    test_array_4 = test_array.clone().detach() + 4
    test_array_5 = test_array.clone().detach() + 5
    batch_tensor_test = torch.zeros((2, 1, 5, 5, 15))
    batch_tensor_test[0, :, :, :, 0: 5] = test_array
    batch_tensor_test[0, :, :, :, 5:10] = test_array_1
    batch_tensor_test[0, :, :, :, 10::] = test_array_2
    batch_tensor_test[1, :, :, :, 0: 5] = test_array_3
    batch_tensor_test[1, :, :, :, 5:10] = test_array_4
    batch_tensor_test[1, :, :, :, 10::] = test_array_5

    test_flatten = flatten_batch(batch_tensor_test, 125)

    print(test_flatten.size())
    print(test_flatten[0, :, :])
    print(test_flatten[1, :, :])
