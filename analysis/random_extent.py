import random
import numpy as np


def random_from_given_probability(candidate_array, candidate_probability, return_shape=None, replace=True):
    """

    :param candidate_array:
    :param candidate_probability:
    :param return_shape:
    :param replace
    :return:
    """
    assert len(candidate_array) == len(candidate_probability)
    return np.random.choice(
        candidate_array, return_shape, p=np.array(candidate_probability) / np.sum(candidate_probability),
        replace=replace)


if __name__ == '__main__':
    test_candidate = [1, 3, 5, 7, 11]
    print(random_from_given_probability(test_candidate, test_candidate, (2, 3)))