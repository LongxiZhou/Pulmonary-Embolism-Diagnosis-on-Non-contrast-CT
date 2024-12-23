import numpy as np


def transformation_on_array(input_array, transformation_flow, reverse=False):
    """

    :param input_array:
    :param transformation_flow: list of operations, or operation
    [
        {"translate": (x, y, ...),},   # will not change shape, use 0 pad
        {"reshape": ((x_old, y_old, ...), (x_new, y_new, ...)),},   # change shape from old to new, only for 2D and 3D
        {"pad_or_crop": (x, y, ...),},  # pad or crop symmetrically. new shape is x_old + 2 * x
    ]

    :param reverse: undo the transformation (there may be small information loss due to reshape, pad/crop)

    array = transformation_on_array(transformation_on_array(array, flow, False), flow, True)

    :return: transformed_array
    """
    if type(transformation_flow) is dict:
        transformation_flow = [transformation_flow]
    if len(transformation_flow) == 0:
        return input_array
    if reverse:
        transformation_flow = list(transformation_flow[::-1])
    for operation in transformation_flow:
        assert len(operation.keys()) == 1
        if list(operation.keys())[0] == "translate":
            input_array = translate_array_old(input_array, operation["translate"], reverse=reverse)
        elif list(operation.keys())[0] == "pad_or_crop":
            input_array = pad_or_crop_array(input_array, operation["pad_or_crop"], reverse=reverse)
        elif list(operation.keys())[0] == "reshape":
            shape_old, shape_new = operation["reshape"]
            if not reverse:
                assert np.shape(input_array) == shape_old
                input_array = reshape_array(input_array, new_shape=shape_new)
            else:
                assert np.shape(input_array) == shape_new
                input_array = reshape_array(input_array, new_shape=shape_old)
        else:
            raise ValueError
    return input_array


def translate_array_old(input_array, translation_vector, reverse=False):
    """

    move the content, use 0 to pad. deep copy the value

    :param input_array:
    :param translation_vector: like (5, 3, 10)
    :param reverse:
    :return: translated_array
    """
    assert len(translation_vector) == len(np.shape(input_array))
    translated_array = np.array(input_array)
    if reverse:
        translation_vector = -np.array(translation_vector)
    for dim, move in enumerate(translation_vector):
        translated_array = translate_array_on_dim(translated_array, int(move), dim)
    return translated_array


def reshape_array(input_array, new_shape):
    """
    only for 2D and 3D

    :param input_array:
    :param new_shape: (x_new, y_new, ...)
    :return: array in new shape in dtype float32 or float64
    """
    from format_convert.spatial_normalize import rescale_to_new_shape
    return rescale_to_new_shape(input_array, new_shape, change_format=True)


def pad_or_crop_array(input_array, operation_vector, reverse=False):
    """

     use 0 to pad

    :param input_array:
    :param operation_vector: (x, y, ...),  # pad or crop symmetrically. new shape is x_old + 2 * x
    :param reverse:
    :return:
    """

    if reverse:
        operation_vector = -np.array(operation_vector, 'int32')

    original_shape = np.shape(input_array)
    pad_shape = []

    slice_tuple_pad = []
    slice_tuple_crop = []

    assert len(operation_vector) == len(original_shape)
    for i, dim in enumerate(original_shape):
        if operation_vector[i] <= 0:
            assert -operation_vector[i] * 2 <= original_shape[i]
            pad_shape.append(dim)
            slice_tuple_pad.append(slice(None))
            slice_tuple_crop.append(slice(-operation_vector[i], dim + operation_vector[i]))
        else:
            pad_shape.append(int(dim + 2 * operation_vector[i]))
            slice_tuple_pad.append(slice(operation_vector[i], dim + operation_vector[i]))
            slice_tuple_crop.append(slice(None))

    slice_tuple_pad = tuple(slice_tuple_pad)
    slice_tuple_crop = tuple(slice_tuple_crop)

    transformed_array = np.zeros(pad_shape, input_array.dtype)
    transformed_array[slice_tuple_pad] = input_array  # pad
    transformed_array = transformed_array[slice_tuple_crop]  # crop
    return transformed_array


def translate_array_on_dim(input_array, move, dim=0):
    """

    move the content along the given dim, use 0 to pad.
    Deep copy the array

    :param input_array:
    :param move:
    :param dim:
    :return: translated_array
    """
    assert type(move) is int and type(dim) is int
    shape_array = np.shape(input_array)
    assert 0 <= dim < len(shape_array)
    assert abs(move) < shape_array[dim]
    if move == 0:
        return np.array(input_array)

    translated_array = input_array * 0

    if move > 0:
        for i in range(move, shape_array[dim]):
            slice_object = []
            for j in range(dim):
                slice_object.append(slice(None))
            slice_object.append(i)
            slice_object = tuple(slice_object)
            translated_array[slice_object] = input_array[(slice(None),) * dim + (i - move,)]
    else:
        move = -move
        for i in range(move, shape_array[dim]):
            slice_object = []
            for j in range(dim):
                slice_object.append(slice(None))
            slice_object.append(i)
            slice_object = tuple(slice_object)
            translated_array[(slice(None),) * dim + (i - move,)] = input_array[slice_object]

    return translated_array


def translate_array(input_array, translation_vector, reverse=False):
    """

    move the content, use 0 to pad. deep copy the value

    :param input_array:
    :param translation_vector: like (5, 3, 10)
    :param reverse:
    :return: translated_array
    """
    shape_array = input_array.shape
    assert len(translation_vector) == len(shape_array)

    if reverse:
        translation_vector = -np.array(translation_vector)

    translated_array = input_array * 0

    def get_slice_on_dim(move, length):
        if move >= 0:
            slice_original = slice(0, length - move)
            slice_translate = slice(move, length)
        else:
            move = -move
            slice_original = slice(move, length)
            slice_translate = slice(0, length - move)

        return slice_original, slice_translate

    slice_sequence_original = []
    slice_sequence_translate = []

    for dim in range(len(translation_vector)):
        slice_ori, slice_trans = get_slice_on_dim(translation_vector[dim], shape_array[dim])
        slice_sequence_original.append(slice_ori)
        slice_sequence_translate.append(slice_trans)

    slice_sequence_original = tuple(slice_sequence_original)
    slice_sequence_translate = tuple(slice_sequence_translate)

    translated_array[slice_sequence_translate] = input_array[slice_sequence_original]

    return translated_array


def down_sample_central_mass_center_and_crop_size(array_rescaled, crop=False):
    """

    first down sample from (512, 512, 512) to (256, 256, 256), then set mass center to (128, 128, 128),

    then crop to (192, 192, 192) optional

    :param crop:
    :param array_rescaled: in shape (512, 512, 512)
    :return: array in shape (192, 192, 192) or (256, 256, 256), transformation_flow
    """
    from analysis.point_cloud import get_mass_center

    # reshape
    transformation_step_1 = {"reshape": ((512, 512, 512), (256, 256, 256))}
    down_sampled_array = transformation_on_array(array_rescaled, transformation_step_1, reverse=False)

    # translate
    mass_center = get_mass_center(np.where(down_sampled_array > 0), median=True)  # (x, y, z)
    transformation_step_2 = {"translate": (128 - mass_center[0], 128 - mass_center[1], 128 - mass_center[2])}
    down_sampled_array = transformation_on_array(down_sampled_array, transformation_step_2, reverse=False)

    if not crop:
        return down_sampled_array, [transformation_step_1, transformation_step_2]

    # crop
    transformation_step_3 = {"pad_or_crop": (-32, -32, -32)}
    down_sampled_array = transformation_on_array(down_sampled_array, transformation_step_3, reverse=False)

    final_transformation_flow = [transformation_step_1, transformation_step_2, transformation_step_3]
    return down_sampled_array, final_transformation_flow


if __name__ == '__main__':
    random_array = np.random.random((20, 10, 15))

    translated_1 = translate_array_old(random_array, (5, -3, 4))
    translated_2 = translate_array(random_array, (5, -3, 4))
    print(np.sum(np.abs(translated_1 - translated_2)))
    exit()

    slice_sequence = (slice(0, 10), slice(2, 5), 1)
    random_array = np.random.random((20, 10, 15))

    print(np.shape(random_array))

    slice_array_1 = random_array[slice_sequence]
    slice_array_2 = random_array[0: 10, 2: 5, 1]
    print(np.sum(np.abs(slice_array_1 - slice_array_2)))

    exit()
    import Tool_Functions.Functions as Functions
    from Tool_Functions.performance_metrics import dice_score_two_class

    array_vessel = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/lung_mask/'
                           'patient-id-050.npz')['array']
    vessel_processed, flow = down_sample_central_mass_center_and_crop_size(array_vessel)
    print(np.shape(array_vessel))
    print(flow)
    vessel_2 = transformation_on_array(array_vessel, flow)

    vessel_recover = transformation_on_array(vessel_processed, flow, reverse=True)

    vessel_processed_2 = transformation_on_array(vessel_recover, flow)
    vessel_recover_2 = transformation_on_array(vessel_processed_2, flow, reverse=True)

    print(dice_score_two_class(array_vessel, vessel_recover))
    print(dice_score_two_class(vessel_2, vessel_processed))
    print(dice_score_two_class(vessel_recover, vessel_recover_2))

    exit()

    test_image = np.zeros([50, 50], 'float32')
    test_image[10: 15, 15: 25] = 1
    Functions.image_show(test_image)

    test_image_4 = pad_or_crop_array(test_image, (6, -10))
    test_image_5 = pad_or_crop_array(test_image_4, (6, -10), reverse=True)
    np.testing.assert_array_equal(test_image_5, test_image)
    Functions.image_show(test_image_4)
    Functions.image_show(test_image_5)

    test_image_2 = translate_array_old(test_image, (5, 20))
    test_image_3 = translate_array_old(test_image_2, (5, 20), reverse=True)
    Functions.image_show(test_image_2)
    Functions.image_show(test_image_3)
    np.testing.assert_array_equal(test_image_3, test_image)
