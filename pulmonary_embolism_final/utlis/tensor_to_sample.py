"""
given a batch of sample sequence, it will be convert to tensors in shape [B, N, flatten_dim]
the outputs are also tensors in shape [B, N, flatten_dim]

put back the outputs to new key in the batch of sample sequence

example, the model output clot probability tensor in [B, N, flatten_dim]
"""
import numpy as np


def tensor_to_semantic_in_sample_sequence(list_sample_sequence, tensor_flatten, semantic, background=None,
                                          shape_cube=(5, 5, 5)):
    """

    :param list_sample_sequence:
    :param tensor_flatten: tensor or numpy array on CPU
    :param semantic: str
    :param background: None, or float. None means not use this feature, float means if all value in a cube equals
    background, replace with None
    :param shape_cube
    :return: list sample sequence
    """

    # check format
    num_sequences = len(list_sample_sequence)
    array_flatten = np.array(tensor_flatten)
    shape_array = np.shape(array_flatten)
    assert num_sequences == shape_array[0]
    max_length_sequence = 0
    for sample_sequence in list_sample_sequence:
        if len(sample_sequence) > max_length_sequence:
            max_length_sequence = len(sample_sequence)
    assert max_length_sequence == shape_array[1]
    assert shape_cube[0] * shape_cube[1] * shape_cube[2] == shape_array[2]
    assert len(shape_cube) == 3 and len(shape_array) == 3

    for i in range(num_sequences):
        for j in range(list_sample_sequence[i]):
            item = list_sample_sequence[i][j]  # is a dict
            semantic_value = np.reshape(array_flatten[i, j, :], shape_cube)
            if background is not None:
                if np.sum(np.abs(semantic_value - background)) == 0:
                    semantic_value = None
            item[semantic] = semantic_value

    return list_sample_sequence


if __name__ == '__main__':
    exit()
