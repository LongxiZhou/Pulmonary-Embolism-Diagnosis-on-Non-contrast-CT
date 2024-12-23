import numpy as np
import Tool_Functions.Functions as Functions
import cv2


def show_2d_function(func, x_range=(0, 1), y_range=(0, 1), resolution=(1000, 1000), leave_cpu_num=1, show=True):
    resolution_x = resolution[0]
    resolution_y = resolution[1]
    step_x = (x_range[1] - x_range[0])/resolution_x
    step_y = (y_range[1] - y_range[0])/resolution_y
    import multiprocessing as mp
    cpu_cores = mp.cpu_count() - leave_cpu_num
    pool = mp.Pool(processes=cpu_cores)
    locations_x = np.ones([resolution_y, resolution_x], 'float32') * np.arange(x_range[0], x_range[1], step_x)
    locations_y = np.ones([resolution_x, resolution_y], 'float32') * np.arange(y_range[0], y_range[1], step_y)
    locations_y = cv2.flip(np.transpose(locations_y), 0)
    locations = np.stack([locations_x, locations_y], axis=2)
    locations = np.reshape(locations, [resolution_y * resolution_x, 2])
    picture = np.array(pool.map(func, locations), 'float32')
    picture = np.reshape(picture, [resolution_y, resolution_x])
    if show:
        Functions.image_show(picture)
    return picture


def visualize_mask_dataset(top_dict_rescaled_ct, top_dict_mask, image_save_top_dict):
    import os

    fn_list = os.listdir(top_dict_rescaled_ct)

    for name in fn_list:

        print("processing", name)

        if name[-1] == 'z':
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, name))['array']
            semantic_mask = np.load(os.path.join(top_dict_mask, name))['array']
        else:
            rescaled_ct = np.load(os.path.join(top_dict_rescaled_ct, name))
            semantic_mask = np.load(os.path.join(top_dict_mask, name[:-4] + '.npz'))['array']

        bounding_box_z = Functions.get_bounding_box(semantic_mask)[2]

        rescaled_ct = np.clip(Functions.change_to_HU(rescaled_ct), -200, 400)

        for z in range(bounding_box_z[0], bounding_box_z[1]):
            save_path = os.path.join(image_save_top_dict, name[:-4], str(z) + '.png')
            Functions.merge_image_with_mask(rescaled_ct[:, :, z], semantic_mask[:, :, z], show=False,
                                            save_path=save_path, dpi=300)


if __name__ == '__main__':
    visualize_mask_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/CTA/',
                           '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/blood_clot_mask_on_CTA/',
                           '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/blood_clot_gt_rescaled/')