import Tool_Functions.Functions as Functions
import numpy as np
import os


def visualize_gt(dict_rescaled_ct, dict_rescaled_gt, pic_save_dict, z_interval=1, max_visualize=20):
    """

    :param pic_save_dict:
    :param dict_rescaled_ct:
    :param dict_rescaled_gt:
    :param z_interval:
    :param max_visualize:
    :return:
    """
    array_name_list = os.listdir(dict_rescaled_ct)
    for name in array_name_list:
        print(name)
        rescaled_ct = np.load(dict_rescaled_ct + name)
        rescaled_ct_lung_window = np.clip(rescaled_ct, -0.5, 0.5)
        rescaled_gt = np.load(dict_rescaled_gt + name[:-1] + 'z')['array']
        rescaled_gt = np.array(rescaled_gt > 0.5, 'float32')

        z_loc_set = set(np.where(rescaled_gt > 0.5)[2])

        if len(z_loc_set) > max_visualize:
            z_interval = int(len(z_loc_set) / max_visualize)
        z_loc_list = list(z_loc_set)
        z_loc_list.sort()

        for z in z_loc_list[::z_interval]:
            Functions.merge_image_with_mask(rescaled_ct_lung_window[:, :, z], rescaled_gt[:, :, z],
                                            save_path=pic_save_dict + name[:-4] + '_' + str(z) + '.png', show=False)


if __name__ == '__main__':
    visualize_gt('/home/zhoul0a/Desktop/pulmonary nodules/data_v1/rescaled_ct/',
                 '/home/zhoul0a/Desktop/pulmonary nodules/data_v1/rescaled_gt_reduced/',
                 '/home/zhoul0a/Desktop/pulmonary nodules/data_v1/visualization/z_view_reduced/')
