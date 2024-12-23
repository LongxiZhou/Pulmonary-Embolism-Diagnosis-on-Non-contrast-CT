import Tool_Functions.Functions as Functions
import os
import numpy as np


import Tool_Functions.file_operations as file_operations
wrong_list = ['Scanner-B-B21.pickle', 'hq0004_2020-03-15.pickle']

dataset_list = ['/data_disk/chest_ct_direction/training_samples/clip_max_50HU',
                '/data_disk/chest_ct_direction/training_samples/not_clip']

for fn in wrong_list:
    for dataset in dataset_list:
        path_wrong_sample = os.path.join(dataset, fn)
        file_operations.remove_path_or_directory(path_wrong_sample)
exit()


def show_sample_image(sample_image):
    Functions.image_show(np.array(sample_image, 'float32'), gray=True)


top_dict_sample = '/data_disk/chest_ct_direction/training_samples/not_clip'
fn_list = os.listdir(top_dict_sample)

fn_list.sort()

wrong_list = [] # ['Scanner-B-B21.pickle', 'hq0004_2020-03-15']

for i in range(4808, len(fn_list), 1):
    fn = fn_list[i]
    if len(wrong_list) > 0:
        if not fn == wrong_list[-1]:
            continue
        else:
            wrong_list = []

    sample_list = Functions.pickle_load_object(os.path.join(top_dict_sample, fn))
    first_sample = sample_list[0]

    image, class_id = first_sample
    print(i, fn, class_id)
    show_sample_image(image)
