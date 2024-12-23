import cv2
from skimage.filters import frangi
from Tool_Functions.Functions import func_parallel
import classic_models.Unet_3D.slicing_array_to_cubes as cube_array_converter
import numpy as np


def window_transform(img, win_min, win_max):
    for i in range(img.shape[0]):
        img[i] = 255.0*(img[i] - win_min)/(win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255)/img[i].max()
        img[i] = img[i]*c
    return img.astype(np.uint8)


def sigmoid(img, alpha, beta):
    return 1 / (1 + np.exp((beta - img) / alpha))


def vessel_enhancement_single_thread(input_tuple):
    image, label, absolute_location = input_tuple
    if np.sum(label) == 0:
        return None, absolute_location
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    label = cv2.erode(label, kernel)
    roi = image * label
    roi_sigmoid = sigmoid(roi, 30, 60)
    roi_frangi = frangi(roi_sigmoid, sigmas=range(1, 5, 1), alpha=0.25, beta=0.25, gamma=75, black_ridges=False)
    cv2.normalize(roi_frangi, roi_frangi, 0, 1, cv2.NORM_MINMAX)
    positive_roi = sorted(roi_frangi[roi_frangi > 0])
    if len(positive_roi) == 0:
        return None, absolute_location
    thresh = np.percentile(positive_roi, 95)
    if (1 - thresh) == 0:
        return None, absolute_location
    vessel = (roi_frangi - thresh) * (roi_frangi > thresh) / (1 - thresh)
    # print("num vessel voxel:", np.sum(vessel))
    return np.where(vessel > 0), absolute_location  # loc_array for vessels, and the absolute location for this cube


def vessel_enhancement_parallel(image, mask, cube_size=(64, 64, 64), step=(48, 48, 48), num_workers=24):
    """

    :param image: input 3D array in numpy float32
    :param mask: input 3D array in numpy float32
    :param cube_size: split image into cubes
    :param step: None for dense paving of cubes
    :param num_workers: max parallel count
    :return: vessel mask in numpy float32
    """
    assert np.shape(image) == np.shape(mask)
    image = image * 1600 - 600
    image = window_transform(image, -1000.0, 650.0)
    image_cube_list = cube_array_converter.convert_numpy_array_to_cube_sequence(image, cube_size, step)
    mask_cube_list = cube_array_converter.convert_numpy_array_to_cube_sequence(mask, cube_size, step)

    input_list = []

    for i in range(len(image_cube_list)):
        # each item is (image, label, absolute_location)
        input_list.append((image_cube_list[i][0], mask_cube_list[i][0], mask_cube_list[i][1]))
    value_list = func_parallel(vessel_enhancement_single_thread, input_list, parallel_count=num_workers)

    value_cube_list = []
    for item in value_list:
        if item[0] is None:
            continue
        value_cube = np.zeros(cube_size, 'float32')
        value_cube[item[0]] = 1
        value_cube_list.append((value_cube, item[1]))

    return_array = cube_array_converter.convert_cube_sequence_to_numpy_array(value_cube_list, np.shape(mask), mask=True)
    return return_array


if __name__ == '__main__':
    exit()
