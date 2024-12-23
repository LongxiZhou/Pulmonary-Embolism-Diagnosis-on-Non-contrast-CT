from classic_models.Unet_3D.slicing_array_to_cubes import convert_numpy_array_to_cube_sequence
import os
import numpy as np
import Tool_Functions.Functions as Functions


top_dict_load = '/data_disk/artery_vein_project/extract_blood_region/training_data/complete_array/' \
                    'CTA/stack_array_artery/'
top_dict_save = '/data_disk/artery_vein_project/extract_blood_region/training_data/sliced_sample/' \
                'CTA/stack_array_artery/'

fn_list = os.listdir(top_dict_load)
print(len(fn_list))
for fn in fn_list:
    sample_array = np.load(top_dict_load + fn)['array']
    sample_cube_list_0 = convert_numpy_array_to_cube_sequence(sample_array[0], (128, 128, 128))
    sample_cube_list_1 = convert_numpy_array_to_cube_sequence(sample_array[1], (128, 128, 128))
    sample_cube_list_2 = convert_numpy_array_to_cube_sequence(sample_array[2], (128, 128, 128))
    sample_cube_list_3 = convert_numpy_array_to_cube_sequence(sample_array[3], (128, 128, 128))

    sample_cube_list = []
    for index in range(len(sample_cube_list_0)):
        if np.sum(sample_cube_list_0[index][0]) == 0:
            continue
        sample_cube_list.append(
            np.stack((sample_cube_list_0[index][0], sample_cube_list_1[index][0],
                      sample_cube_list_2[index][0], sample_cube_list_3[index][0])))

    print(fn, len(sample_cube_list))

    for index in range(len(sample_cube_list)):
        Functions.save_np_array(
            top_dict_save, fn[:-4] + '_' + str(index) + '.npz', sample_cube_list[index], compress=True)

    """
    import visualization.visualize_3d.visualize_stl as stl
    stl.visualize_numpy_as_stl(sample_cube_list[8][0])
    stl.visualize_numpy_as_stl(sample_cube_list[8][0] - sample_cube_list[8][1])
    """

