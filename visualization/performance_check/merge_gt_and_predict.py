import visualization.visualize_3d.highlight_semantics as highlight_semantics
import numpy as np
import os
import Tool_Functions.Functions as Functions


def visualize_prediction_with_gt(array_data, array_prediction, array_gt, save_dict, array_name=None):
    """

    :param array_data: 3D numpy array in 0-1
    :param array_prediction: 3D numpy array in 0-1
    :param array_gt: 3D numpy array in 0-1
    :param save_dict: directory for the images
    :param array_name: e.g., patient-id_scan-time
    :return:
    """

    t_p = array_gt * array_prediction
    f_p = array_prediction * (1 - array_gt)
    f_n = array_gt * (1 - array_prediction)

    merged_array = highlight_semantics.highlight_mask(f_p, array_data, channel='Y', further_highlight=False)
    merged_array = highlight_semantics.highlight_mask(f_n, merged_array, channel='R', further_highlight=True)
    merged_array = highlight_semantics.highlight_mask(t_p, merged_array, channel='G', further_highlight=True)

    z_to_plot = list(set(np.where((array_gt + array_prediction) > 0.5)[2]))

    if array_name is not None:
        for z in z_to_plot:
            Functions.image_save(merged_array[:, :, z, :], os.path.join(save_dict, array_name) + '_' + str(z),
                                 high_resolution=True)
        return None
    for z in z_to_plot:
        Functions.image_save(merged_array[:, :, z, :], os.path.join(save_dict, str(z)), high_resolution=True)


if __name__ == '__main__':
    exit()
