import numpy as np
import Tool_Functions.Functions as Functions


def get_4d_sincos_pos_embed_loc_list(embed_dim, loc_list):
    """

    :param embed_dim: mod 8 == 0
    :param loc_list: a list with element of locations: [(x, y, z, b), ...]
    :return: the embedding vector with shape [len(loc_list), embed_dim]
    """
    assert embed_dim % 8 == 0
    loc_array = Functions.get_location_array(loc_list)  # shape (4, len(loc_list))

    embedding_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, loc_array[0])  # (len(loc_list), D/4)
    embedding_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, loc_array[1])  # (len(loc_list), D/4)
    embedding_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, loc_array[2])  # (len(loc_list), D/4)
    embedding_b = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, loc_array[3])  # (len(loc_list), D/4)

    pos_embed = np.concatenate([embedding_x, embedding_y, embedding_z, embedding_b], axis=1)  # (len(loc_list), D)

    return pos_embed


def get_3d_sincos_pos_embed_loc_list(embed_dim, loc_list):
    """

    :param embed_dim: mod 6 == 0
    :param loc_list: a list with element of locations: [(x, y, z), ...]
    :return: the embedding vector with shape [len(loc_list), embed_dim]
    """
    assert embed_dim % 6 == 0
    loc_array = Functions.get_location_array(loc_list)  # shape (3, len(loc_list))

    embedding_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, loc_array[0])  # (len(loc_list), D/3)
    embedding_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, loc_array[1])  # (len(loc_list), D/3)
    embedding_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, loc_array[2])  # (len(loc_list), D/3)

    pos_embed = np.concatenate([embedding_x, embedding_y, embedding_z], axis=1)  # (len(loc_list), D)

    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)

    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.array(pos, 'float32')
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':

    li = [(0, 0, 1), (0, 0, -1), (0, -1, 1), (0, 1, -1)]
    embedding = get_3d_sincos_pos_embed_loc_list(192, li)
    embedding_2 = get_4d_sincos_pos_embed_loc_list(192, li)

    print(np.sum(embedding))
    print(np.sum(np.abs(embedding - embedding_2)))
    exit()

    li = [(0, 0, 0) for i in range(5)] + [(1, 1, 1) for i in range(5)]

    embedding = get_3d_sincos_pos_embed_loc_list(18, li)

    embedding = np.reshape(embedding, [2, 5, 18])
    print(embedding[0, :, :])
    print(embedding[1, :, :])

    exit()
    embedding = get_2d_sincos_pos_embed(16, 10)
    print(np.shape(embedding))
    for i in range(20):
        print(embedding[i, :])
    exit()
