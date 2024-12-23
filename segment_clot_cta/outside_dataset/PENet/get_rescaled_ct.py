import os
import numpy as np
import Tool_Functions.Functions as Functions


def process_fn(file_name):
    """

    pad or crop to (512, 512, 512)

    :param file_name: in .npy (format in PENet)
    :return:
    """
    save_path = os.path.join(target_dict, file_name[:-4] + '.npz')
    if os.path.exists(save_path):
        print("processed")
        return None

    original_array = np.load(os.path.join(source_dict, file_name))
    original_array = np.clip(original_array, -1000, 1000)
    original_array = np.transpose(original_array, (1, 2, 0))

    original_array = (original_array + 600) / 1600

    shape_original = np.shape(original_array)
    assert shape_original[0] == 512 and shape_original[1] == 512

    rescaled_array = np.zeros((512, 512, 512), 'float16')
    if shape_original[2] < 512:
        pad = 256 - round(shape_original[2] / 2)
        rescaled_array[:, :, pad: pad + shape_original[2]] = original_array
    else:
        crop = int(shape_original[2] / 2) - 256
        rescaled_array[:, :, :] = original_array[:, :, crop: crop + 512]

    Functions.save_np_array(target_dict, file_name[:-4] + '.npz', rescaled_array, compress=True)


if __name__ == '__main__':
    source_dict = '/data_disk/Altolia_share/PENet_dataset/PENet_original'
    target_dict = '/data_disk/Altolia_share/PENet_dataset/rescaled_ct'

    current_fold = (0, 2)

    fn_list = os.listdir(source_dict)[current_fold[0]::current_fold[1]]
    processed_count = 0
    for fn in fn_list:
        print("processing:", fn, processed_count, '/', len(fn_list))
        process_fn(fn)
        processed_count += 1
