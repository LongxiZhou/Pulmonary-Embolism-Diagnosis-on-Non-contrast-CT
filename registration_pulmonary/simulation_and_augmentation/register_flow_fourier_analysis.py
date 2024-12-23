"""

registration flow = fx * i + fy * j + fz * j, in shape (fx, fy, fz)
"""


import numpy as np
import Tool_Functions.Functions as Functions


def add_sin_cos_component_array(array, axis, num_pi, amplitude, sin=True):
    """

    array[...axis...][index] += amplitude * sin(num_pi * pi * index / (array.shape[axis] - 1))
    adding value will be broadcast the other axis

    :param sin: sin or cos
    :param array: float numpy array
    :param axis:
    :param num_pi:
    :param amplitude:

    :return: array added the sin on the axis
    """
    shape_on_axis = np.shape(array)[axis]
    added_vector = np.arange(0, shape_on_axis)
    added_vector = added_vector * np.pi * num_pi / (shape_on_axis - 1)

    if sin:
        added_vector = amplitude * np.sin(added_vector)
    else:
        added_vector = amplitude * np.cos(added_vector)

    new_shape = [shape_on_axis, ]
    for i in range(len(np.shape(array)) - 1):
        new_shape.append(1)
    added_vector = np.reshape(added_vector, new_shape)

    array = np.swapaxes(array, 0, axis)
    array = array + added_vector
    array = np.swapaxes(array, 0, axis)

    return array


if __name__ == '__main__':
    import torch
    import visualization.visualize_3d.visualize_stl as stl
    from analysis.center_line_and_depth_3D import get_surface_distance

    flow_x = np.zeros([100, 100, 100], 'float32')
    flow_y = np.zeros([100, 100, 100], 'float32')
    flow_z = np.zeros([100, 100, 100], 'float32')
    flow_x = add_sin_cos_component_array(flow_x, 0, 4, 5, sin=True)
    flow_y = add_sin_cos_component_array(flow_y, 1, 4, 5, sin=True)
    flow_z = add_sin_cos_component_array(flow_z, 2, 4, 5, sin=True)

    registration_flow = np.stack((flow_x, flow_y, flow_z), axis=0)
    registration_flow = np.reshape(registration_flow, (1, 3, 100, 100, 100))
    registration_flow = torch.FloatTensor(registration_flow)

    image_cube = np.zeros([100, 100, 100], 'float32')
    image_cube[25: 75, 25: 75, 25: 75] = 1

    image_cube = get_surface_distance(image_cube, strict=True)

    image_cube_tensor = torch.FloatTensor(image_cube)
    image_cube_tensor = torch.reshape(image_cube_tensor, (1, 1, 100, 100, 100))

    from registration_pulmonary.models.model_no_landmark import RegisterWithGivenFlow

    plot_flow = RegisterWithGivenFlow(size=(100, 100, 100))

    modified_tensor = plot_flow(image_cube_tensor, registration_flow)
    modified = modified_tensor.detach().numpy()[0, 0]

    Functions.image_show(image_cube[:, :, 50])
    Functions.image_show(modified[:, :, 50])
    exit()

    stl.visualize_numpy_as_stl(image_cube)
    stl.visualize_numpy_as_stl(modified)

    exit()
    Functions.image_show(flow_x[:, 50, :])
    exit()
