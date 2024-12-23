import random
import format_convert.basic_transformations as basic_transformations


def set_fixed_and_registered_as_the_same(batch_sample):
    fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = batch_sample

    moving_image_tensor = fixed_image_tensor

    return fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list


def apply_translate_augmentation(batch_sample, params):
    """

    :param batch_sample:
    :param params:
    :return: batch_sample modified
    """
    fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = batch_sample

    move_lower_bound, move_higher_bound = params["random_translate_voxel_range"]

    # image_tensor with shape [B, N, L, L, L], so translation_vector is (0, 0, a, b, c)
    translate_vector = [0, 0]
    for i in range(3):
        translate_vector.append(random.randint(move_lower_bound, move_higher_bound))

    moving_image_tensor = basic_transformations.translate_array(moving_image_tensor, translate_vector, reverse=False)

    # make padding
    fixed_image_tensor = basic_transformations.translate_array(fixed_image_tensor, translate_vector, reverse=False)
    fixed_image_tensor = basic_transformations.translate_array(fixed_image_tensor, translate_vector, reverse=True)

    return fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list
