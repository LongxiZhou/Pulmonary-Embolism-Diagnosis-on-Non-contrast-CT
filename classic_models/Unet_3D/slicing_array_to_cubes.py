import numpy as np
import Tool_Functions.Functions as Functions


def convert_numpy_array_to_cube_sequence(numpy_array, cube_size, step=None):
    """
    slice the numpy array into cube sequence
    all data in numpy array will be included
    changes in the cube will not affect the numpy_array

    :param numpy_array: numpy float32 array with shape [x, y, z]
    :param cube_size: tuple like (64, 64, 64)
    :param step: tuple like (64, 64, 64)
    :return: a list, each item: (cube, (x_start, y_start, z_start))
    """
    array_shape = np.shape(numpy_array)
    assert len(cube_size) == 3 and len(array_shape) == 3
    if step is None:
        step = cube_size
        
    temp_array = np.ones((int(array_shape[0] / step[0]), 
                          int(array_shape[1] / step[1]), 
                          int(array_shape[2] / step[2])), 'float32')
    loc_array = np.where(temp_array > 0.5)
    loc_array = (loc_array[0] * step[0], loc_array[1] * step[1], loc_array[2] * step[2])
    list_start_locations = Functions.get_location_list(loc_array)  # where we extract the cube

    return_list = []
    for location in list_start_locations:
        x_start, y_start, z_start = location
        x_end = min(x_start + cube_size[0], array_shape[0])
        y_end = min(y_start + cube_size[1], array_shape[1])
        z_end = min(z_start + cube_size[2], array_shape[2])
        cube = np.zeros(cube_size, 'float32')
        cube[0: (x_end - x_start), 0: (y_end - y_start), 0: (z_end - z_start)] = \
            numpy_array[x_start: x_end, y_start: y_end, z_start: z_end]
        return_list.append((cube, location))
    return return_list


def convert_cube_sequence_to_numpy_array(list_cubes_and_loc, shape, mask=False):
    """

    :param mask:
    :param list_cubes_and_loc: same format with the return of "convert_numpy_array_to_cube_sequence"
                                each item: (cube, (x_start, y_start, z_start))
    :param shape: the shape of the return array
    :return: numpy float32 array
    """
    assert len(shape) == 3
    return_array = np.zeros(shape, 'float32')

    for item in list_cubes_and_loc:
        cube, (x_start, y_start, z_start) = item
        cube_shape = np.shape(cube)
        x_end = min(x_start + cube_shape[0], shape[0])
        y_end = min(y_start + cube_shape[1], shape[1])
        z_end = min(z_start + cube_shape[2], shape[2])
        if not mask:
            return_array[x_start: x_end, y_start: y_end, z_start: z_end] = \
                cube[0: (x_end - x_start), 0: (y_end - y_start), 0: (z_end - z_start)]
        else:
            return_array[x_start: x_end, y_start: y_end, z_start: z_end] = \
                cube[0: (x_end - x_start), 0: (y_end - y_start), 0: (z_end - z_start)] + \
                return_array[x_start: x_end, y_start: y_end, z_start: z_end]

        if mask:
            return_array = np.clip(return_array, 0, 1)
    return return_array


if __name__ == '__main__':

    exit()
    test_array = np.load('/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/CTA/patient-id-021.npy')
    test_slicing = convert_numpy_array_to_cube_sequence(test_array, (64, 64, 64), (25, 25, 25))
    new_array = convert_cube_sequence_to_numpy_array(test_slicing, (512, 512, 512))
    Functions.image_show(new_array[:, :, 256])
    import basic_tissue_prediction.predict_rescaled as predictor
    import visualization.visualize_3d.visualize_stl as stl
    test_lung = predictor.predict_lung_masks_rescaled_array(new_array)
    stl.visualize_numpy_as_stl(test_lung)
    exit()
