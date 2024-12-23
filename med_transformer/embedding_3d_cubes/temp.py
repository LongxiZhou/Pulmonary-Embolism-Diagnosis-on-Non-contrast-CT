import torch
import torch.nn as nn
import os
import numpy as np
import Tool_Functions.Functions as Functions
from skimage import measure


exit()
test_array = np.load('/home/zhoul0a/Downloads/RAD-ChestCT_dataset/trn04556.npz')['ct']
print(np.shape(test_array))
Functions.image_show(test_array[156, :, :])
exit()

array = np.zeros([10, 10, 10])

array_2 = np.zeros([10, 10, 10])

loc_list = [(1, 2, 3), (3, 3, 3), (1, 1, 1), (3, 5, 7), (7, 3, 2)]

for loc in loc_list:
    array_2[loc] = 1


print(np.where(array_2 == 1))
print(Functions.get_location_array(loc_list))
array[Functions.get_location_array(loc_list)] = 1

print(np.sum(array))

print(np.sum(np.abs(array_2 - array)))
exit()
print(Functions.get_location_array(loc_list))
print(np.where(array_2 == 1))

exit()

print(np.shape(array[0:2, 4:5, :]))
exit()
array[(1, 1, 1)] = 1
array[np.where(array == 1)] = 2
print(array)
exit()
top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/pickle_dataset/blood_vessel_merge/'
fn_list = os.listdir(top_dict)

length_list = []

for fn in fn_list:
    samples = Functions.pickle_load_object(top_dict + fn)
    length_list.append(len(samples))

length_list.sort()
print(length_list)
print(np.average(length_list), np.median(length_list), np.quantile(length_list, 0.8))
print()

exit()


def directory_for_path(path):
    """
    change path or directory to the directory
    :param path: like /home/zhoul0a/Desktop/hospitalize_data_dict.pickle
    :return: the directory for the path, like /home/zhoul0a/Desktop/
    """
    assert len(path) > 0
    if not path[0] == '/':
        path = '/' + path
    if path[-1] == '/':
        return path
    name_list = path.split('/')[:-1]
    print(name_list)
    print(len(name_list))
    current_path = name_list[0]
    for file_name in name_list[1::]:
        current_path = os.path.join(current_path, file_name)
    if not current_path[0] == '/':
        current_path = '/' + current_path
    return current_path + '/'


print(directory_for_path('/home/zhoul0a/Desktop/'))

exit()
input_tensor = torch.randn(2, 3, 11, 11, 11)

m = nn.Conv3d(3, 128, (6, 6, 6), dilation=(2, 2, 2))

m2 = nn.Conv3d(3, 128, (5, 5, 5), dilation=(2, 2, 2))

print(m2(input_tensor[:, :, 1:-1, 1:-1, 1:-1]).size())
