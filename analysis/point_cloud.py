import math
import Tool_Functions.Functions as Functions
import numpy as np


def rotate_point_cloud(point_cloud, rotate_matrix):
    """

    :param point_cloud: location list, like [(x, y, z), ] or numpy array with size n * 3
    :param rotate_matrix
    :return: rotated point cloud, in form like [(x, y, z), ]
    """
    point_cloud_array = np.array(point_cloud)
    assert len(np.shape(point_cloud_array)) == 2
    point_cloud_array = np.swapaxes(point_cloud_array, 0, 1)

    point_cloud_array_rotated = np.matmul(rotate_matrix, point_cloud_array)

    return Functions.get_location_list(list(point_cloud_array_rotated))


def get_rotate_matrix_vector_degree_3d(axis_vector, rotate_degree):
    """

    :param axis_vector: (x, y, z), the rotation axis is the vector from (0, 0, 0) to (x, y, z)
    :param rotate_degree: degree of counter-clock wise rotate
    :return:
    """

    rotate_degree = rotate_degree / 180 * math.pi

    assert len(axis_vector) == 3
    axis_normal = np.array(axis_vector)
    x, y, z = axis_normal / math.sqrt(np.sum(np.square(axis_normal)))

    cos_d = math.cos(rotate_degree)
    sin_d = math.sin(rotate_degree)

    rotate_matrix = np.array(
        [[cos_d + (1 - cos_d) * x * x, (1 - cos_d) * x * y - sin_d * z, (1 - cos_d) * x * z + sin_d * y],
         [(1 - cos_d) * x * y + sin_d * z, cos_d + (1 - cos_d) * y * y, (1 - cos_d) * y * z - sin_d * x],
         [(1 - cos_d) * x * z - sin_d * y, (1 - cos_d) * y * z + sin_d * x, cos_d + (1 - cos_d) * z * z]])

    return rotate_matrix


def set_mass_center(loc_list_or_array, new_mass_center=None, show=True, median=True, int_loc_arrays=True):
    """
    set the mass center to the given point (deep copy)
    :param int_loc_arrays: True, translate vector will be int
    :param median: use median or average for the mass center
    :param show:
    :param loc_list_or_array:
    :param new_mass_center:
    :return: same format with the input
    """
    if type(loc_list_or_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_array)
    else:
        loc_array = loc_list_or_array

    dimension = len(loc_array)
    if new_mass_center is None:
        new_mass_center = list(np.zeros((dimension, ), 'int32'))
    assert len(new_mass_center) == dimension

    if len(loc_array[0]) == 0:
        return loc_list_or_array

    old_mass_center = get_mass_center(loc_list_or_array, median=median)

    if int_loc_arrays:
        translate_vector = []
        for dim in range(dimension):
            translate_vector.append(int(new_mass_center[dim] - old_mass_center[dim]))

        new_loc_array = translate_point_cloud(loc_array, translate_vector)
        if show:
            print("old mass center is:", old_mass_center)
            print("new mass center is:", get_mass_center(new_loc_array, median=median))

        return new_loc_array

    new_loc_array = []

    for dim in range(dimension):
        new_loc_array.append(np.array(loc_array[dim]) - old_mass_center[dim] + new_mass_center[dim])

    if show:
        print("old mass center is:", old_mass_center)
        print("new mass center is:", get_mass_center(new_loc_array, median=median))

    if loc_list_or_array is list:
        return Functions.get_location_list(new_loc_array)
    return tuple(new_loc_array)


def get_mass_center(loc_list_or_array, median=True, cast_to_int=False):
    """
    get the mass center for the given point cloud
    :param cast_to_int: return int
    :param median: use median or average for the mass center
    :param loc_list_or_array:
    :return: (x, y, z, ...)
    """
    if type(loc_list_or_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_array)
    else:
        loc_array = loc_list_or_array

    mass_center = []
    for loc_projection_array in loc_array:
        if median:
            mass_center.append(np.median(loc_projection_array))
        else:
            mass_center.append(np.average(loc_projection_array))

    if cast_to_int:
        mass_center_new = []
        for value in mass_center:
            mass_center_new.append(round(value))
        return tuple(mass_center_new)

    return tuple(mass_center)


def translate_point_cloud(loc_list_or_array, translate_vector):
    """
    translate (deep copy)
    :param translate_vector: the vector for moving every point like (1, 2, 3, 4)
    :param loc_list_or_array:
    :return: same format with the input
    """
    convert_to_list = False
    if type(loc_list_or_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_array, dtype='float32')
        convert_to_list = True
    else:
        loc_array = loc_list_or_array

    dimension = len(loc_array)
    assert len(translate_vector) == dimension

    new_loc_array = []
    for dim in range(dimension):
        new_loc_array.append(np.array(loc_array[dim]) + translate_vector[dim])

    if convert_to_list:
        return Functions.get_location_list(new_loc_array)
    return tuple(new_loc_array)


def point_cloud_to_numpy_array(loc_list_or_array, value_array=None, pad=2):
    """

    form the array for the given point_cloud

    :param loc_list_or_array:
    :param value_array: the value for each point
    :param pad:
    :return: float32 numpy array in shape [x_max - x_min + 1 + pad * 2, ...]
    """
    if type(loc_list_or_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_array)
    else:
        loc_array = loc_list_or_array

    if value_array is None:
        value_array = np.ones(np.shape(loc_array[0]), 'float32')
    else:
        assert len(value_array) == len(loc_array[0])

    return_array_shape = []
    mass_center = []
    original_mass_center = get_mass_center(loc_array, True)
    for dim, projection_array in enumerate(loc_array):
        dim_max = np.max(projection_array)
        dim_min = np.min(projection_array)
        range_dim = dim_max - dim_min + 1 + 2 * pad
        return_array_shape.append(int(range_dim))
        mass_center.append(original_mass_center[dim] - dim_min + pad)

    return_array = np.zeros(return_array_shape, 'float32')

    new_loc_array = set_mass_center(loc_array, mass_center, median=True, show=False)

    int_loc_array = []
    for projection_array in new_loc_array:
        int_loc_array.append(np.array(projection_array, 'int32'))

    return_array[tuple(int_loc_array)] = value_array
    return return_array


if __name__ == '__main__':
    test_image = np.zeros([10, 10])
    test_image[1:5, 3: 6] = 1
    loc_list_test = Functions.get_location_list(np.where(test_image > 0))

    print(set_mass_center(loc_list_test))
    exit()

    Functions.image_show(point_cloud_to_numpy_array(loc_list_test))
    exit()
    list_loc_array = Functions.pickle_load_object('/home/zhoul0a/Desktop/pulmonary_embolism/clot_simulation/'
                                                  'simulated_results/lesion/'
                                                  'list-of-loc-array_surface-growth_volume_5000-50000.pickle')
    Functions.show_point_cloud_3d(Functions.get_location_list(list_loc_array[0]))

    exit()
    test = np.load(
        '/data_disk/rescaled_ct_and_semantics/depth_and_center-line/healthy_people/four_center_data/airway_center_line/'
        'Scanner-A-A1.npz')['array']
    point_cloud_test = np.where(test > 0.5)
    print(get_mass_center(point_cloud_test))
    print(get_mass_center(translate_point_cloud(point_cloud_test, (10, -10, 5))))
    exit()
    print(get_mass_center(Functions.get_location_list(point_cloud_test), median=False))
    exit()
    Functions.show_point_cloud_3d(Functions.get_location_list(set_mass_center(point_cloud_test)))
    Functions.show_point_cloud_3d(Functions.get_location_list(point_cloud_test))
    exit()
    point_cloud_test = Functions.get_location_list(point_cloud_test)
    point_cloud_rotated = rotate_point_cloud(point_cloud_test, get_rotate_matrix_vector_degree_3d((0, 0, 1), 90))
    Functions.show_point_cloud_3d(point_cloud_rotated + point_cloud_test)
