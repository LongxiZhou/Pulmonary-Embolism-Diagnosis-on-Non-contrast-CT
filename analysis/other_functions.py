import Tool_Functions.Functions as Functions
import numpy as np
import format_convert.spatial_normalize as spatial_normalize
import analysis.center_line_and_depth_3D as get_depth
import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.connectivity_refine_fast as get_sorted_connect_regions


def smooth_mask(binary_mask, surface_add=0):
    for surface in range(surface_add + 1):
        binary_mask = binary_mask + get_surface.get_surface(binary_mask, outer=True, strict=False)
    binary_mask = binary_mask - get_surface.get_surface(binary_mask, outer=False, strict=False)
    return binary_mask


def get_left_and_right_lung(lung_mask, down_sample_ratio=4, smooth_intensity=1):
    """

    :param lung_mask: numpy array
    :param down_sample_ratio: int
    :param smooth_intensity: can pad holes with radius smooth_intensity * 2
    :return: lung_mask, -1 left lung, 0 not lung, 1 right lung, in float32
    """
    lung_mask_smoothed = np.array(smooth_mask(lung_mask, surface_add=smooth_intensity), 'float32')
    lung_mask_down_sampled = lung_mask_smoothed[::down_sample_ratio, ::down_sample_ratio, ::down_sample_ratio]

    depth_array, max_depth = get_depth.get_surface_distance(
        lung_mask_down_sampled, strict=True, return_max_distance=True)

    temp_array = np.zeros(np.shape(lung_mask_down_sampled), 'int8')

    depth = 1
    while True:
        loc_array = np.where(depth_array == depth)
        temp_array[loc_array] = 1
        id_loc_dict = get_sorted_connect_regions.get_sorted_connected_region_fast(
            temp_array, strict=False, show=False, sort=True)

        if not len(id_loc_dict) == 2 and depth < max_depth:
            depth += 1
            temp_array[loc_array] = 0
        else:
            break

    if depth == max_depth:
        raise ValueError("cannot distinguish two lungs, max depth:", max_depth)

    loc_list_1 = id_loc_dict[1]
    loc_list_2 = id_loc_dict[2]

    loc_array_1 = Functions.get_location_array(loc_list_1)
    loc_array_2 = Functions.get_location_array(loc_list_2)

    mass_center_left = [np.median(loc_array_1[0]), np.median(loc_array_1[1]), np.median(loc_array_1[2])]
    mass_center_right = [np.median(loc_array_2[0]), np.median(loc_array_2[1]), np.median(loc_array_2[2])]

    if not mass_center_left[1] < mass_center_right[1]:
        mass_center_left, mass_center_right = mass_center_right, mass_center_left

    def whether_close_to_left(location):
        distance_to_left = (location[0] - mass_center_left[0]) ** 2 + (location[1] - mass_center_left[1]) ** 2 + (
                location[2] - mass_center_left[2]) ** 2

        distance_to_right = (location[0] - mass_center_right[0]) ** 2 + (location[1] - mass_center_right[1]) ** 2 + (
                location[2] - mass_center_right[2]) ** 2

        return distance_to_left > distance_to_right

    lung_mask_down_sampled = lung_mask_down_sampled + get_surface.get_surface(
        lung_mask_down_sampled, outer=True, strict=False)

    loc_list = Functions.get_location_list(np.where(lung_mask_down_sampled > 0))
    for loc in loc_list:
        if whether_close_to_left(loc):
            lung_mask_down_sampled[loc] = -1
        else:
            lung_mask_down_sampled[loc] = 1

    return spatial_normalize.rescale_to_new_shape(
        lung_mask_down_sampled, np.shape(lung_mask), change_format=False) * lung_mask


if __name__ == '__main__':
    lung_test = Functions.load_nii('/home/zhoul0a/Desktop/transfer/temp/lung_mask.nii.gz')
    lung_test = np.array(lung_test > 0, 'float32')
    new_lung = get_left_and_right_lung(lung_test, down_sample_ratio=4, smooth_intensity=1)

    for i in range(0, 40, 3):
        Functions.image_show(new_lung[:, :, i])
