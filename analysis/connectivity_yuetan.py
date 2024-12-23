import numpy as np
from skimage.measure import label as connect
import skimage.measure as measure


def get_key(d, value):
    return [k for k, v in d.items() if v == value]


def select_region(mask, num, thre=50, bias=False):
    center = np.array(mask.shape) / 2
    # print(center)
    labels, nums = connect(mask, connectivity=None, return_num=True)
    prop = measure.regionprops(labels)
    label_sum = {}
    # print(nums)
    new_mask = np.zeros(np.shape(mask), 'float32')
    for label in range(nums):
        if prop[label].area > thre:
            # if prop[label].area < 150:
            label_sum[label + 1] = prop[label].area
    # print(label_sum)
    area_list = []
    for value in label_sum.values():
        area_list.append(value)
    area_list = np.sort(area_list)
    area_list = area_list[::-1]
    # print(area_list)
    if num >= len(area_list):
        num = len(area_list)
    for i in range(num):
        area = area_list[i]
        label = get_key(label_sum, area)[0]
        # visualize_numpy_as_stl(np.array(labels == label, "float32"))
        section = np.array(labels == label, "float32")
        if bias:
            section_center = compute_center(section)
            # print(section_center)
            center_distance = compute_distance(center, section_center)
            # print(center_distance)
            if center_distance < 100:
                new_mask += section
        else:
            new_mask += section

    return new_mask


def compute_distance(point_1, point_2):
    distance = (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2 + (point_1[2] - point_2[2]) ** 2
    return np.sqrt(distance)


def compute_center(array):
    point_loc = np.where(array != 0)

    num = np.sum(array)
    x_center = np.sum(point_loc[0])
    y_center = np.sum(point_loc[1])
    z_center = np.sum(point_loc[2])

    return x_center / num, y_center / num, z_center / num


def random_select_region(mask, thre=0.5, bias=False):
    mask = np.array(mask > 0.5, "float32")
    center = np.array(mask.shape) / 2
    # print(center)
    labels, nums = connect(mask, connectivity=2, return_num=True)
    prop = measure.regionprops(labels)
    label_sum = {}
    # print(nums)
    new_mask = mask * 0
    for label in range(nums):
        rand = np.random.uniform(0, 1)
        if rand < thre:
            # if prop[label].area < 150:
            label_sum[label + 1] = prop[label].area
    # print(label_sum)
    area_list = []
    for value in label_sum.values():
        area_list.append(value)
    area_list = np.sort(area_list)
    area_list = area_list[::-1]
    # print(area_list)
    num = len(area_list)
    # print(num)
    for i in range(num):
        # print(i)
        area = area_list[i]
        label = get_key(label_sum, area)[0]
        # visualize_numpy_as_stl(np.array(labels == label, "float32"))
        section = np.array(labels == label, "float32")
        if bias:
            section_center = compute_center(section)
            # print(section_center)
            center_distance = compute_distance(center, section_center)
            # print(center_distance)
            if center_distance < 100:
                new_mask += section
        else:
            new_mask += section

    return new_mask

