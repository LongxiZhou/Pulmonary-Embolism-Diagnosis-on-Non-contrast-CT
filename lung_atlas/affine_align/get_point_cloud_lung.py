"""
See function: sample_surface_point_cloud
Input: binary mask for the lung in float32 numpy array, number of cloud_point
Output: a list of locations in [(x, y, z), ...]

The idea for sampling is to let the point cloud evenly distributed on the surface
"""
import Tool_Functions.Functions as Functions
import numpy as np
import math
import analysis.get_surface_rim_adjacent_mean as get_surface


def sample_surface_point_cloud(binary_mask, number_sample_points):
    """

    :param binary_mask: three dimension numpy float32 array
    :param number_sample_points:
    :return: list of tuples, each tuples is the location for (x, y, z)
    """
    assert len(np.shape(binary_mask)) == 3
    assert number_sample_points > 0

    surface = get_surface.get_surface(binary_mask, False, True)

    total_surface_points = np.sum(surface)
    print("there are", total_surface_points, 'surface voxels,', 'sample', number_sample_points,
          'sample ratio', number_sample_points / total_surface_points)

    loc_array = np.where(surface > 0.5)
    location_list_surface = list(zip(loc_array[0], loc_array[1], loc_array[2]))

    def compare_func(location_a, location_b):
        """
        z-axis is the first to compare, then x-axis, then y-axis, as asymmetry z > x > y for lung masks
        :param location_a: (x, y, z)
        :param location_b: (x, y, z)
        :return: -1, 0 or 1
        """
        if location_a[2] > location_b[2]:
            return 1
        if location_a[2] < location_b[2]:
            return -1
        if location_a[0] > location_b[0]:
            return 1
        if location_a[0] < location_b[0]:
            return -1
        if location_a[1] > location_b[1]:
            return 1
        if location_a[1] < location_b[1]:
            return -1
        return 0

    location_list_surface_sorted = Functions.customized_sort(location_list_surface, compare_func, reverse=False)

    sample_interval = math.ceil(total_surface_points / number_sample_points)

    sample_location_list = list(location_list_surface_sorted[::sample_interval])

    # evenly select extra points to reach the total number
    extra_points = number_sample_points - len(sample_location_list)

    if extra_points == 0:
        return sample_location_list

    extra_interval = int(total_surface_points / extra_points)
    further_added = 0
    while extra_points - further_added > 0:
        sample_location_list.append(location_list_surface_sorted[further_added * extra_interval])
        further_added += 1

    print(len(sample_location_list))
    return sample_location_list


if __name__ == '__main__':
    lung_mask = np.load('/home/zhoul0a/Desktop/absolutely_normal/masks/lung_masks/Scanner-C_B39.npz')['array']
    location_list = sample_surface_point_cloud(lung_mask, 2000)
    location_array = list(zip(*location_list))

    import matplotlib.pyplot as plt
    x = location_array[0]
    y = location_array[1]
    z = location_array[2]
    # fig = plt.figure()
    ax = plt.gca(projection="3d")
    ax.scatter(x, y, z, s=1)
    plt.show()
    plt.close()
