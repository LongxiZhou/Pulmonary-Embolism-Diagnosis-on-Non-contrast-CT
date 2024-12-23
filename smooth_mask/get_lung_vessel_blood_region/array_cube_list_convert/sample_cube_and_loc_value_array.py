import numpy as np


def convert_sample_cube_to_loc_value_info(sample_cube):
    """

    :param sample_cube: in shape (4, x, y, z)
    :return: [(x, y, z), info_channel_0, info_channel_1, info_channel_2, info_channel_3]
    """
    shape = np.shape(sample_cube)
    return_list = [shape[1::]]
    for channel in range(shape[0]):
        return_list.append(get_info_channel(sample_cube[channel]))
    return return_list


def get_info_channel(array):

    loc_array = np.where(array > 0)  # >, <, == is a simple instruct; != is 4 times slower
    value_array = array[loc_array]

    return loc_array, value_array


def convert_loc_value_info_to_sample_cube(loc_value_info):
    shape = [len(loc_value_info) - 1]
    shape = shape + list(loc_value_info[0])
    return_array = np.zeros(shape, 'float32')
    channel_info_list = loc_value_info[1::]
    for channel in range(shape[0]):
        return_array[channel][channel_info_list[channel][0]] = channel_info_list[channel][1]
    return return_array


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    exit()
    test_array = np.load('/data_disk/artery_vein_project/extract_blood_region/training_data/'
                         'array_version/CTA/stack_array_artery/AL00001.npz')['array']
    test_info_version = convert_sample_cube_to_loc_value_info(test_array)

    new_test_array = convert_loc_value_info_to_sample_cube(test_info_version)

    print(np.sum(np.abs(new_test_array - test_array)))
    Functions.image_show(new_test_array[0, :, :, 128])
    exit()
    import time

    test_array = np.zeros([1000, 1000, 1000], 'float32')
    start = time.time()
    np.where(test_array > 1)
    end = time.time()
    print(end - start)
