import SimpleITK as sitk
import bintrees
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os


def read_in_mha(path):
    """
    Convert the .mha to .npy. Annotation should be binary.
    :param path: the file path for the .mha file of the annotation
    :return: the binary numpy array for the annotation
    """
    ar = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0, 'int32')
    return mask


def load_dicom(path, show=False, specify_name=None):
    # return freq numpy array of the dicom file, and the slice number
    if show:
        content = pydicom.read_file(path)
        print(content)
        # print(content['ContentDate'])

    ds = sitk.ReadImage(path)

    img_array = sitk.GetArrayFromImage(ds)

    if specify_name is not None:
        for name in specify_name:
            print(pydicom.read_file(path)[name].value)

    #  frame_num, width, height = img_array.shape

    return img_array[0, :, :], pydicom.read_file(path)['InstanceNumber'].value


def stack_dcm_files_simplest(dic, show=True):
    """
    Convert the .dicom files into a numpy array
    :param dic: the directory that stories the .dicom files for a CT scan
    :param show: print out information of the dicom files
    :return: the numpy array stacked according to the .dicom files
    """

    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    if show:
        print("number_dcm_files:", num_slices)
    first_slice = load_dicom(os.path.join(dic, dcm_file_names[0]))[0]

    first_content = pydicom.read_file(os.path.join(dic, dcm_file_names[0]))
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    if show:
        print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    slice_id_list = []
    array_3d = np.zeros([rows, columns, num_slices + 3], 'float32')
    for file in dcm_file_names:
        data_array, slice_id = load_dicom(os.path.join(dic, file))
        slice_id_list.append(slice_id)
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)

        array_3d[:, :, num_slices - slice_id] = data_array
    assert np.max(slice_id_list) - np.min(slice_id_list) + 1 == len(slice_id_list)
    array_3d = array_3d[:, :, num_slices - np.max(slice_id_list): num_slices - np.min(slice_id_list) + 1]

    return array_3d


def linear_value_change(array, min_value, max_value, data_type='float32'):
    # linearly cast to [min_value, max_value]
    max_original = np.max(array) + 0.000001
    min_original = np.min(array)
    assert max_value > min_value
    assert max_original > min_original
    return_array = np.array(array, data_type)
    return_array -= min_original
    return_array = return_array / ((max_original - min_original) * (max_value - min_value)) + min_value
    return return_array


def image_save(picture, path, gray=False, high_resolution=False, dpi=None):
    save_dict = path[:-len(path.split('/')[-1])]
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    picture = linear_value_change(picture, 0, 1)
    if not gray:
        plt.cla()
        plt.axis('off')
        plt.imshow(picture)
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            return None
        if high_resolution:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=600)
        else:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight')
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], 'float32')
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            return None
        if high_resolution:
            plt.cla()
            plt.axis('off')
            plt.imshow(gray_img)
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=600)
        else:
            plt.cla()
            plt.imshow(gray_img)
            plt.savefig(path)
    return None


def merge_two_picture(picture, mask, color='R'):
    # picture is freq 2-d array, mask is also freq 2-d array
    picture = linear_value_change(picture, 0, 1)
    mask = linear_value_change(mask, 0, 1)

    a = np.shape(picture)[0]
    b = np.shape(picture)[1]
    assert np.shape(picture) == np.shape(mask)
    output = np.zeros([a, b * 2, 3], 'float32')
    output[:, 0:b, 0] = picture
    output[:, 0:b, 1] = picture
    output[:, 0:b, 2] = picture
    if color == 'R':
        output[:, b::, 0] = picture + mask
        output[:, b::, 1] = picture - mask
        output[:, b::, 2] = picture - mask
    if color == 'G':
        output[:, b::, 0] = picture - mask
        output[:, b::, 1] = picture + mask
        output[:, b::, 2] = picture - mask
    if color == 'B':
        output[:, b::, 0] = picture - mask
        output[:, b::, 1] = picture - mask
        output[:, b::, 2] = picture + mask
    output = np.clip(output, 0, 1)
    return output


def merge_image_with_mask(image, mask_image, save_path=None, high_resolution=True, color='R'):
    merged_image = np.array(image)
    merged_image = merge_two_picture(merged_image, mask_image, color=color)

    if save_path is not None:
        image_save(merged_image, save_path, high_resolution=high_resolution)

    return merged_image


def center_loc(input_mask, axis_list=None):
    """

    :param input_mask: binary
    :param axis_list: which axis to return, None for all_file axis
    :return: a list for the center of the axis
    """
    total_axis = len(np.shape(input_mask))
    if axis_list is None:
        axis_list = list(np.arange(total_axis))

    center_list = []
    loc_positive = np.where(input_mask > 0.5)

    for axis in axis_list:
        center_list.append(int(np.average(loc_positive[axis])))
    return center_list


def check_dcm_and_mha_for_one_scan(dcm_dic, mha_path, image_save_path):
    """
    Generate the image for checking the annotation quality
    :param dcm_dic: directory for the .dicom files of the CT scan
    :param mha_path: path for the annotation .mha file
    :param image_save_path: path for the image for check the annotation quality
    :return: None
    """
    ct_data = stack_dcm_files_simplest(dcm_dic)
    ct_data_lung_window = np.clip(ct_data, -1400, 200)
    annotation = read_in_mha(mha_path)
    z_center = center_loc(annotation)[2]
    merge_image_with_mask(ct_data_lung_window[:, :, z_center], annotation[:, :, z_center], save_path=image_save_path)
    return None


if __name__ == '__main__':
    check_dcm_and_mha_for_one_scan(
        '/home/zhoul0a/Desktop/vein_artery_identification/dcm_gt_artery_vein/f032/2020-03-10/Data/raw_data',
        '/home/zhoul0a/Desktop/vein_artery_identification/dcm_gt_artery_vein/f032/2020-03-10/Data/ground_truth/xb.mha',
        '/home/zhoul0a/Desktop/picture/temp/f032_2020-03-10_xb.png')
