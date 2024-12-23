import copy
import numpy as np
import time
import pickle
import math
import matplotlib.pyplot as plt
import shutil
from scipy import io
import imutils
from medpy import io
import imageio
import os
import random
import nibabel as nib
import cv2
import pydicom
import SimpleITK as sitk
import multiprocessing as mp
import hashlib

np.set_printoptions(suppress=True)


def get_father_dict(path_or_dict=None):
    if path_or_dict is None:
        return os.path.abspath(os.path.join(os.getcwd(), '..'))
    name_list = path_or_dict.split('/')
    valid_name_list = []
    for folder_name in name_list:
        if len(folder_name) == 0:
            continue
        valid_name_list.append(folder_name)

    new_path = '/'

    for folder_name in valid_name_list[:-1]:
        new_path = os.path.join(new_path, folder_name)
    return new_path


def strip_suffix(path_or_file_name):
    """

    example:
    input file_name return file_name
    input file_name. return file_name
    input file_name.npz return file_name
    input file_name.npz.npz return file_name.npz

    :param path_or_file_name:
    :return: path_or_file_name removed suffix
    """
    reversed_path = path_or_file_name[::-1]
    index = 0
    already_stripped = False
    for char in reversed_path:
        if char == '.':
            break
        if char == '/':
            already_stripped = True
            break
        index += 1
    if already_stripped:
        return reversed_path[::-1]
    if index == len(reversed_path):
        return reversed_path[::-1]
    index += 1
    return reversed_path[::-1][:-index]


def get_suffix(path_or_file_name):
    stripped_suffix = strip_suffix(path_or_file_name)
    return path_or_file_name[len(strip_suffix(path_or_file_name)):]


def iteration_with_time_bar(iterable_object):
    """

    :param iterable_object: like list, range(a, b, c), etc.
    :return: modified iterable object: print progress bar for each step
    """
    import tqdm
    return tqdm.tqdm(iterable_object)


def show_memory(variable, unit='KB'):
    from sys import getsizeof
    scale = {
        'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    memory = getsizeof(variable) / scale
    print(memory, unit)


def load_dicom(path, show=False, specify_name=None, load_wit_sitk=True, first_slice=True):
    # return a numpy array of the dicom file, and the slice number
    if show:
        content = pydicom.read_file(path, force=True)
        print(content)

    if specify_name is not None:
        for name in specify_name:
            print(pydicom.read_file(path)[name].value)

    if load_wit_sitk:
        ds = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(ds)
        if first_slice:
            img_array = img_array[0]
        return img_array, pydicom.read_file(path)['InstanceNumber'].value
    else:
        content = pydicom.read_file(path, force=True)
        img_array = content.pixel_array * content.RescaleSlope + content.RescaleIntercept
        return img_array, content['InstanceNumber'].value


def get_dicom_resolution(path):
    first_content = pydicom.read_file(path)
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    return resolutions


def dicom_and_prediction(prediction, dicom_image, ww_wc=(1600, -600)):
    dicom_image = dicom_image - ww_wc[1]
    dicom_image = dicom_image / ww_wc[0]
    dicom_image = dicom_image + 0.5

    shape = np.shape(dicom_image)
    merge = np.zeros((shape[0], shape[1] * 2, 3), 'float32')
    merge[:, 0: shape[1], 0] = dicom_image
    merge[:, 0: shape[1], 1] = dicom_image
    merge[:, 0: shape[1], 2] = dicom_image
    merge[:, shape[1]::, 0] = dicom_image
    merge[:, shape[1]::, 1] = dicom_image - prediction
    merge[:, shape[1]::, 2] = dicom_image - prediction
    return np.clip(merge, 0, 1)


def convert_png_to_np_array(file_path):
    return imageio.imread(file_path)


def rotate_and_scale_image(image_array, angle, center=None, scale=1.0):
    """

    :param image_array: numpy array in (x, y), (x, y, 3) or (x, y, 4)
    :param angle: range 0, 360, like rotate 90 change x pos to y pos
    :param center: the center location of rotate
    :param scale: scale after rotate
    :return: numpy array in same shape of the input
    """
    return imutils.rotate(image_array, angle, center, scale)


def copy_file(source_path, destiny_path):
    # will overwrite is destiny path exist
    shutil.copyfile(source_path, destiny_path)


def extract_wc_ww(value):
    try:
        return int(value)
    except:
        try:
            return int(value[0])
        except:
            print('wc_ww strange')
            exit(1)


def wc_ww(path):
    info = pydicom.read_file(path)
    wc = info['WindowCenter'].value
    ww = info['WindowWidth'].value
    return extract_wc_ww(wc), extract_wc_ww(ww)


def get_median_loc(binary_array):
    loc_array = np.where(binary_array > 0)
    num_dim = len(loc_array)
    if not num_dim > 0:
        return None
    num_voxel = len(loc_array[0])
    if not num_voxel > 0:
        return None

    median_loc = []
    for dim in range(num_dim):
        median_loc.append(round(np.median(loc_array[dim])))
    return median_loc


def array_stat(array, non_zero=False, show=True):
    if show:
        print('array has shape:', np.shape(array))
    if non_zero:
        zero_loc_array = np.where(array == 0)
        mask_non_zero = np.ones(np.shape(array), 'float32')
        mask_non_zero[zero_loc_array] = 0
        non_zero_loc_array = np.where(mask_non_zero == 1)
        array = array[non_zero_loc_array]
        if show:
            print("non zero value count:", len(array))
    min_v, ave, max_v, std = np.min(array), np.average(array), np.max(array), np.std(array)
    if show:
        print('min-average-max:', min_v, ave, max_v)
        print('std:', std)
    return min_v, ave, max_v, std


def sort_non_zero_voxels(array, reverse=False, sparse=False):
    """

    :param sparse: whether the non-zero is sparse, if non-zero < 0.3 total_voxel, set True will be faster
    :param array:
    :param reverse: False means from small to large
    :return: a flatten array sorted from small to large, and the number of non_zero voxels.
    """
    flatten_array = np.reshape(np.array(array), (-1, ))

    if sparse:
        non_zero_locations = np.where(flatten_array != 0)
        sub_array = np.array(flatten_array[0: len(non_zero_locations[0])])
        for i in range(len(non_zero_locations[0])):
            sub_array[i] = flatten_array[non_zero_locations[0][i]]
        if not reverse:
            sub_array.sort()
            return sub_array, len(sub_array)
        sub_array = -sub_array
        sub_array.sort()
        return -sub_array, len(sub_array)

    zero_locations = np.where(flatten_array == 0)

    if not reverse:
        max_value = np.max(flatten_array)

        flatten_array[zero_locations] = max_value + 1
        flatten_array.sort()
        return flatten_array, len(flatten_array) - len(zero_locations[0])

    flatten_array = -flatten_array
    max_value = np.max(flatten_array)
    flatten_array[zero_locations] = max_value + 1
    flatten_array.sort()
    return -flatten_array, len(flatten_array) - len(zero_locations[0])


def load_nii(path):
    # return a numpy array of this .nii or .nii.gz file
    return nib.load(path).get_data()


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


def pickle_save_object(save_path, object_being_save, buffer_path='/home/zhoul0a/Desktop/transfer/buffer_file_longxi',
                       use_buffer=True):
    """
    :param use_buffer:
    :param save_path: like /home/zhoul0a/Desktop/hospitalize_data_dict.pickle
    :param object_being_save: like a dictionary
    :param buffer_path: avoid incomplete files. the buffer, when save is finished, move buffer to the save_path
    :return: None
    """
    name_list = save_path.split('/')[:-1]
    current_path = name_list[0]
    for file_name in name_list[1::]:
        current_path = os.path.join(current_path, file_name)
    if not current_path[0] == '/':
        current_path = '/' + current_path
    if not os.path.exists(current_path):
        os.makedirs(current_path)

    if use_buffer:

        buffer_path = buffer_path + str(hash(save_path)) + str(time.time())

        with open(buffer_path, 'wb') as handle:
            pickle.dump(object_being_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        shutil.move(buffer_path, save_path)

    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(object_being_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_object(save_path):
    with open(save_path, 'rb') as handle:
        return pickle.load(handle)


def save_np_as_nii_gz(array, diction, file_name):
    # save a numpy array as .nii.gz or .nii file
    # e.g. diction = '/home/Desktop/test/' file_name = 'hello.nii.gz', then we have a '/home/Desktop/test/hello.nii.gz'
    # note: np.array_equal(load_nii(diction+file_name), array) is True
    if not os.path.exists(diction):
        os.makedirs(diction)
    nii_file = nib.Nifti1Image(array, np.eye(4))
    nii_file.to_filename(os.path.join(diction, '%s' % file_name))


def save_np_as_mha(np_array, save_dict, file_name, spacing=(1, 1, 1)):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if file_name[-4::] == '.mha':
        file_name = file_name[:-4]
    np_array = np.transpose(np_array, (1, 0, 2))
    """
    
    np_array[np_array < 0.5] = 0
    np_array[np_array >= 0.5] = 1
    np_array = np_array.astype("uint8")
    """
    header = io.Header(spacing=spacing)
    print("mha file path:", os.path.join(save_dict, file_name) + '.mha')
    io.save(np_array, os.path.join(save_dict, file_name) + '.mha', hdr=header, use_compression=True)


def rescale_2d_array(input_array, new_shape_tuple):
    """
    :param input_array: a 2d image array with float32
    :param new_shape_tuple: the output shape, i.e., np.shape(return array)
    :return: the shape normalized array
    """
    assert len(np.shape(input_array)) == 2 and len(new_shape_tuple) == 2
    shape_normalize = cv2.resize(input_array, (new_shape_tuple[1], new_shape_tuple[0]), cv2.INTER_AREA)
    return shape_normalize


def copy_file_or_dir(source_path, save_path, show=True):
    """
    :param show
    :param source_path: path of the
    :param save_path:
    :return:
    """
    if show:
        print("copy path:", source_path)
        print("save_path:", save_path)

    if not os.path.isdir(source_path):
        father_dict = get_father_dict(save_path)
        if os.path.exists(father_dict):
            assert os.path.isdir(father_dict)
        else:
            os.makedirs(father_dict)
        shutil.copyfile(source_path, save_path)
    else:
        if os.path.exists(save_path):
            assert len(os.listdir(save_path)) == 0
            shutil.rmtree(save_path)
        shutil.copytree(source_path, save_path)


def save_np_array(save_dict, file_name, np_array, compress=False, dtype=None, use_buffer=True,
                  buffer_path='/home/zhoul0a/Desktop/transfer/buffer_file_longxi/', save_path=None, **kwargs):
    # if the save_dict not exist, we make the dict

    if save_path is not None:
        assert type(save_path) is str
        save_dict = get_father_dict(save_path)
        file_name = save_path.split('/')[-1]

    if dtype is not None:
        np_array = np.array(np_array, dtype)

    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if len(file_name) >= 4:
        if file_name[-4] == '.':
            file_name = file_name[:-4]
    if compress:
        file_name = file_name + '.npz'
    else:
        file_name = file_name + '.npy'

    save_path = os.path.join(save_dict, file_name)

    if np_array is None:
        print("None input for save path:", save_path)
        return None

    if use_buffer:

        buffer_path = buffer_path + str(hash(file_name)) + str(time.time())
        if compress:
            buffer_path = buffer_path + '.npz'
        else:
            buffer_path = buffer_path + '.npy'

        if not compress:
            np.save(buffer_path, np_array)
        else:
            np.savez_compressed(buffer_path, array=np_array, **kwargs)

        shutil.move(buffer_path, save_path)

    else:
        if not compress:
            np.save(save_dict + file_name, np_array)
        else:
            np.savez_compressed(save_dict + file_name, array=np_array, **kwargs)


def f1_sore_for_binary_mask(prediction, ground_truth, threshold=0.5):
    prediction = np.array(prediction > threshold, 'float32')
    ground_truth = np.array(ground_truth > threshold, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    return 2 * over_lap / (np.sum(prediction) + np.sum(ground_truth))


def get_rim(mask):
    # mask is the mask file which is a [a, b] np array
    # return the rim of the input np array

    a = np.shape(mask)[0]
    b = np.shape(mask)[1]

    return_array = np.zeros([a, b], dtype='int32')

    for i in range(1, a - 1):
        for j in range(1, b - 1):
            if mask[i, j] != 0:
                if mask[i - 1, j] == 0:
                    return_array[i, j] = 1
                    return_array[i - 1, j] = 1
                if mask[i + 1, j] == 0:
                    return_array[i, j] = 1
                    return_array[i + 1, j] = 1
                if mask[i, j - 1] == 0:
                    return_array[i, j] = 1
                    return_array[i, j - 1] = 1
                if mask[i, j + 1] == 0:
                    return_array[i, j] = 1
                    return_array[i, j + 1] = 1

    return return_array


def image_show(picture_in, gray=False, color_bar=False):
    if len(np.shape(picture_in)) == 3:
        if np.shape(picture_in)[2] == 4:
            picture_in = np.array(picture_in, 'int32')
    if not gray:
        plt.imshow(picture_in)
        if color_bar:
            plt.colorbar()
        plt.show()
        return picture_in
    picture = np.array(picture_in, 'float32')
    picture = picture - np.min(picture)
    picture = picture / (np.max(picture) + 0.00000001)
    s = np.shape(picture)
    image = np.zeros([s[0], s[1], 3], 'float32')
    image[:, :, 0] = picture
    image[:, :, 1] = picture
    image[:, :, 2] = picture
    image_show(image, False, color_bar=color_bar)
    return image


def show_multiple_images(height=9, gray=False, clim=(None, None), **kwargs):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(kwargs),
        figsize=(height * len(kwargs), height)
    )
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        if gray:
            pcm = ax.imshow(v, cmap="gray", clim=clim)
        else:
            pcm = ax.imshow(v, clim=clim)
        fig.colorbar(pcm, ax=ax)
        ax.set_title(k)
    fig.tight_layout()
    plt.show()


def merge_two_picture(picture, mask, color='R'):
    # picture is a 2-d array, mask is also a 2-d array
    picture = cast_to_0_1(picture)
    mask = cast_to_0_1(mask)

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


def merge_image_with_mask(image, mask_image, convert_to_rim=False, save_path=None, show=True,
                          high_resolution=True, color='R', dpi=None):
    temp = np.array(image)

    if convert_to_rim:
        rim_array = get_rim(mask_image)
        temp = merge_two_picture(temp, rim_array, color=color)
    else:
        temp = merge_two_picture(temp, mask_image, color=color)

    if save_path is not None:
        image_save(temp, save_path, high_resolution=high_resolution, dpi=dpi)

    if show:
        image_show(temp)

    return temp


def image_save(picture, path, gray=False, high_resolution=False, dpi=600, color_bar=False, remove_axis=True):
    save_dict = path[:-len(path.split('/')[-1])]
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    if not color_bar:
        picture = linear_value_change(picture, 0, 1)
    if not gray:
        plt.cla()
        if remove_axis:
            plt.axis('off')
        plt.imshow(picture)
        if color_bar:
            plt.colorbar()
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            plt.close()
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
            plt.cla()
            if remove_axis:
                plt.axis('off')
            plt.imshow(gray_img)
            if color_bar:
                plt.colorbar()
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            plt.close()
            return None
        if high_resolution:
            plt.cla()
            if remove_axis:
                plt.axis('off')
            plt.imshow(gray_img)
            if color_bar:
                plt.colorbar()
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=600)
        else:
            plt.cla()
            plt.imshow(gray_img)
            if color_bar:
                plt.colorbar()
            plt.savefig(path)
    plt.close()
    return None


def cast_to_0_1(input_array):
    # rescale the input array into range (0, 1)
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    out_array = np.array((input_array - min_value) * 1.0, 'float32')
    out_array = out_array / (max_value - min_value + 0.00001)
    return out_array


def show_signal_distribution(input_array, roi_mask=None, number_data_points=10, save_path=None):

    if roi_mask is not None:
        loc_array = np.where(roi_mask > 0)
        value_array = input_array[loc_array]
    else:
        value_array = np.reshape(input_array, (-1, ))

    value_array.sort()  # from small to large

    max_value, min_value = value_array[-1], value_array[0]

    if max_value == min_value:
        print("all value is the same, equals to:", max_value)

    array_stat(value_array)

    plt.hist(value_array, bins=number_data_points)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def show_data_points(x, y, save_path=None, data_label='data points', x_name='x_axis', y_name='y_axis', title='scatter',
                     ylim=None, xlim=None):
    plt.close()

    if data_label is not None:
        # plot1 = plt.plot(x, y, 'r', label='data points')
        plot1 = plt.plot(x, y, '*', label=data_label)
    else:
        # plot1 = plt.plot(x, y, 'r')
        plot1 = plt.plot(x, y, '*')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法
    if ylim is not None:  # (y_min, y_max)
        plt.ylim(ylim)

    if xlim is not None:  # (x_min, x_max)
        plt.xlim(xlim)

    plt.title(title)

    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600
        plt.savefig(save_path)
        plt.close()


def derivative(func, args, precision=0.000000000001):
    # func = func(args)
    # returns a array of d(func)/d(args) at given args, with error = O(precision)
    # if the third order derivative = 0, then error = 0
    h = math.sqrt(abs(precision))
    num_args = len(args)
    return_list = []
    for i in range(num_args):
        if i > 4:
            return_list.append(0)
            continue
        args[i] += h
        ahead = func(args)
        args[i] -= 2 * h
        behind = func(args)
        args[i] += h
        return_list.append((ahead - behind) / 2 / h)

    return np.array(return_list, 'float32')


def shuffle_array(input_array):
    slices = np.shape(input_array)[0]
    all_indices = np.arange(0, slices, 1)
    random.shuffle(all_indices)
    return_array = input_array[all_indices, :, :, :]
    return return_array


def linear_fit(x, y, show=True):
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - a * sx) / N
    r = (sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    if show:
        print("the fitting result is: y = %10.5f x + %10.5f , r = %10.5f" % (a, b, r))
    return a, b, r


def scale_free_check(scale_list, frequency, cache=10, show=True, remove_front_cache=1, pic_save_path=None, label=None):
    # scale_list is a ordered list recording the measurements, like area, degree, etc
    # auency is a list recording the auency or probability of each scale
    scale_list = np.array(scale_list)
    frequency = np.array(frequency)
    length = len(scale_list)
    assert len(scale_list) == len(frequency)
    if show:
        print("the length of the list is", length)
    step = round(length / cache)

    def get_center(sub_list_scale, sub_list_frequency):
        return sum(sub_list_scale * sub_list_frequency) / sum(sub_list_frequency)

    center_list = []
    total_frequency_list = []
    for loc in range(0, length, step):
        if loc + step >= length:
            end = length
        else:
            end = loc + step
        list_cache_scale = scale_list[loc: end]
        list_cache_frequency = frequency[loc: end]
        center_list.append(get_center(list_cache_scale, list_cache_frequency))
        total_frequency = np.sum(list_cache_frequency)
        if total_frequency == 0:
            print("detect 0 frequency, replace with 1")
            total_frequency = 1
        total_frequency_list.append(total_frequency)
    if show:
        show_data_points(np.log(center_list[remove_front_cache::]), np.log(total_frequency_list[remove_front_cache::]),
                         data_label=label, x_name='log(area)', y_name='log(frequency)', title='scale free check',
                         save_path=pic_save_path)
    return linear_fit(np.log(center_list[remove_front_cache::]), np.log(total_frequency_list[remove_front_cache::]))


def scale_free_check_cdf(scale_list, frequency, show=True, remove_front_point=0, log_p_min=-4):
    # scale_list is a ordered list recording the measurements, like area, degree, etc
    # auency is a list recording the auency or probability of each scale

    cdf = np.zeros(len(frequency))  # cumulative distribution function
    total_count = np.sum(frequency)
    current_cum = 0
    for i in range(len(frequency)):
        current_cum += frequency[i] / total_count
        cdf[i] = current_cum

    comp_cdf = 1 - cdf  # complimentary cumulative distribution function

    largest_observed = 0
    for i in range(len(frequency)):
        if frequency[i] > 0:
            largest_observed = i

    log_com_cdf_list = np.log(comp_cdf[: largest_observed - 1])  # thus every element is > 0

    final_point = largest_observed - 2
    for i in range(largest_observed - 2, 0, -1):
        if log_com_cdf_list[i] > log_p_min:
            final_point = i
            break

    log_k_list = np.log(scale_list)

    a, b, r = \
        linear_fit(log_k_list[remove_front_point: final_point], log_com_cdf_list[remove_front_point: final_point])

    if show:
        show_data_points(log_k_list[remove_front_point: final_point], log_com_cdf_list[remove_front_point: final_point])
        print("scale factor =", -a + 1)

    return -a + 1, r


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


def sigmoid(array, a, b):
    # linearly cast to [0, 1], sigmoid, then linearly cast to [min(array), max(array)]
    min_original = np.min(array)
    max_original = np.max(array)
    return_array = linear_value_change(array, 0, 1)  # cast to [0, 1]
    assert a > 0 and b > 0
    return_array = 1 / (1 + a * np.exp(-b * return_array))  # sigmoid
    return_array = linear_value_change(return_array, min_original, max_original)  # cast to [min(array), max(array)]
    return return_array


def func_parallel(func, list_inputs, leave_cpu_num=1, parallel_count=None):
    """
    :param parallel_count:
    :param func: func(list_inputs[i])
    :param list_inputs: each element is the input of func
    :param leave_cpu_num: num of cpu that not use
    :return: [return_of_func(list_inputs[0]), return_of_func(list_inputs[1]), ...]
    """

    cpu_cores = mp.cpu_count() - leave_cpu_num
    if parallel_count is not None:
        cpu_cores = parallel_count
    pool = mp.Pool(processes=cpu_cores)
    list_outputs = pool.map(func, list_inputs)
    pool.close()
    return list_outputs


def rescaled_to_hu(array):
    return array * 1600 - 600


def read_in_mha(path):
    ar = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0, 'int32')
    return mask  # (x, y, z)


def get_bounding_box(mask, pad=0):
    # mask is a binary array
    # return a list [(x_min, x_max), (y_min, y_max), ...] which is the bounding box of each dimension
    bounding_box = []
    positive_locs = np.where(mask > 0)

    for loc_array in positive_locs:
        min_loc = int(np.min(loc_array) - pad)
        max_loc = int(np.max(loc_array) + pad)
        bounding_box.append((min_loc, max_loc))
    return bounding_box


def get_mass_center_for_binary(mask, median=True, cast_to_int=False):
    # return (x, y, z, ...) same data type with input "mask"
    from analysis.point_cloud import get_mass_center

    loc_array = np.where(mask > 0)

    return get_mass_center(loc_array, median=median, cast_to_int=cast_to_int)


def get_max_projection(mask, projection_dim):
    total_dim = len(np.shape(mask))
    remove_dim = []
    for i in range(total_dim):
        if i == projection_dim:
            continue
        remove_dim.append(i)
    slice_area_array = np.sum(mask, axis=tuple(remove_dim))
    print(np.shape(slice_area_array))
    max_count = -np.inf
    max_index = 0
    for i in range(0, len(slice_area_array)):
        if slice_area_array[i] > max_count:
            max_count = slice_area_array[i]
            max_index = i
    return max_count, max_index


def change_to_HU(input_array):
    return (input_array * 1600) - 600


def deep_copy(input_object):

    return copy.deepcopy(input_object)


def change_to_rescaled(input_array):
    return (input_array + 600) / 1600


def customized_sort(list_like, compare_func, reverse=False):
    """

    :param reverse:
    :param list_like: iterative object
    :param compare_func: takes two element, a, b as input, return -1 or 1. If a > b return 1, other wise return -1
    If reverse is False, the sort is Increasing.
    :return:
    """

    from functools import cmp_to_key
    list_like.sort(key=cmp_to_key(compare_func), reverse=reverse)
    return list_like


def stat_on_mask(reference_array, mask, remove_outliers=0.2):
    """
    stat on the given mask
    :param remove_outliers: e.g. removes largest 20% and smallest 20%
    :param reference_array: like a 3D CT data
    :param mask: like airway mask, binary value
    :return: value mean, std on of the reference_array value on mask
    """
    locations = np.where(mask > 0)
    num_voxels = len(locations[0])
    value_list = []
    for i in range(num_voxels):
        value_list.append(reference_array[locations[0][i], locations[1][i], locations[2][i]])
    value_list.sort()
    assert remove_outliers < 0.5
    value_list = value_list[int(num_voxels * remove_outliers): num_voxels - 2 - int(num_voxels * remove_outliers)]
    value_list = np.array(value_list)
    return np.median(value_list), np.std(value_list)


def get_sorted_values_from_given_region(reference_array, mask, large_mask=False):
    """

    :param reference_array: like a 3D CT data
    :param mask: like airway mask, binary value
    :param large_mask: if the mask is very large, e.g., a fix proportion compared to the reference_array, set it to True
    :return: sorted array with values from the mask region, shaped (np.sum(mask), ) from small to large
    """
    if not large_mask:
        locations = np.where(mask > 0)
        num_voxels = len(locations[0])
        value_list = []
        for i in range(num_voxels):
            value_list.append(reference_array[locations[0][i], locations[1][i], locations[2][i]])
        value_list.sort()
        value_array = np.array(value_list)
        return value_array
    else:

        mask_bounding_box = get_bounding_box(mask, pad=0)
        tight_mask = mask[mask_bounding_box[0][0]: mask_bounding_box[0][1],
                          mask_bounding_box[1][0]: mask_bounding_box[1][1],
                          mask_bounding_box[2][0]: mask_bounding_box[2][1]]
        sub_array = reference_array[mask_bounding_box[0][0]: mask_bounding_box[0][1],
                                    mask_bounding_box[1][0]: mask_bounding_box[1][1],
                                    mask_bounding_box[2][0]: mask_bounding_box[2][1]]

        # deep copy to avoid change to original data
        sub_array = np.array(sub_array, 'float32')
        sub_array = sub_array + (np.max(sub_array) + 1) * tight_mask

        num_mask_points = int(np.sum(tight_mask))

        sub_array = np.reshape(sub_array, (-1,))
        sub_array.sort()
        return sub_array[(len(sub_array) - num_mask_points):]


def find_index_for_value_in_sorted_object(sorted_iterable, value, start=0):
    """
    :param sorted_iterable:
    :param value:
    :param start: index start for recursive use, do not input value for it
    :return: index, int for smallest index that sorted_iterable[index] == value;
                    float for sorted_iterable[round(index)] closest to value
                    None for value not in the range of sorted_iterable
    """
    num_values = len(sorted_iterable)

    extreme_value_a, extreme_value_b = sorted_iterable[0], sorted_iterable[-1]
    if not (extreme_value_a <= value <= extreme_value_b or extreme_value_b <= value <= extreme_value_a):
        return None

    if num_values <= 2:
        if num_values == 2:
            if sorted_iterable[0] == value:
                return start
            if sorted_iterable[1] == value:
                return start + 1

            differ_left = abs(value - sorted_iterable[0])
            differ_right = abs(value - sorted_iterable[1])

            if differ_right < differ_left:
                return start + 0.75
            return start + 0.25
        return None

    def merge_two(input_a, input_b):
        if input_a is None and input_b is None:
            return None
        if input_a is not None and input_b is None:
            return input_a
        if input_a is None and input_b is not None:
            return input_b
        return min(input_a, input_b)

    left_part = sorted_iterable[0: int(num_values / 2) + 1]
    right_part = sorted_iterable[int(num_values / 2):]

    return merge_two(find_index_for_value_in_sorted_object(left_part, value, start),
                     find_index_for_value_in_sorted_object(right_part, value, start + int(num_values / 2)))


def rename_path(old_path, new_path):
    os.rename(old_path, new_path)


def split_dict_and_name(path):
    name = path.split('/')[-1]
    return path[0: len(path) - len(name)], name


def save_np_to_path(save_path, np_array):
    save_dict, file_name = split_dict_and_name(save_path)
    assert file_name[-4::] == '.npy' or file_name[-4::] == '.npz'
    if file_name[-1] == 'z':
        save_np_array(save_dict, file_name, np_array, compress=True)
    else:
        save_np_array(save_dict, file_name, np_array, compress=False)


def get_heat_map(cam_map, target_shape=None):
    # input a numpy array with shape (a, b)
    min_value, max_value = np.min(cam_map), np.max(cam_map)
    cam_map = (cam_map - min_value) / (max_value + 0.00001) * 255
    cam_map = np.array(cam_map, 'int32')
    if target_shape is not None:
        assert len(target_shape) == 2

        cam_map = cv2.resize(np.array(cam_map, 'float32'), target_shape)  # must in float to resize
    colored_cam = cv2.normalize(cam_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_cam = cv2.applyColorMap(colored_cam, cv2.COLORMAP_JET)

    return_image = np.zeros(np.shape(colored_cam), 'int32')
    return_image[:, :, 0] = colored_cam[:, :, 2]
    return_image[:, :, 1] = colored_cam[:, :, 1]
    return_image[:, :, 2] = colored_cam[:, :, 0]

    return return_image / 255


def merge_with_heat_map(data_image, cam_map, signal_rescale=False):
    """

    :param signal_rescale: 0-1 rescale of data_image
    :param data_image: a numpy array with shape (a, b) or (a, b, 3)
    :param cam_map: a numpy array with shape (c, d)
    :return: merged image with shape (a, b, 3), in float32, min 0 max 1.0
    """
    shape_image = np.shape(data_image)
    if not shape_image == np.shape(cam_map):
        heat_map = get_heat_map(cam_map, target_shape=(shape_image[0], shape_image[1]))
    else:
        heat_map = get_heat_map(cam_map, target_shape=None)
    if signal_rescale:
        min_value, max_value = np.min(data_image), np.max(data_image)
        data_image = (data_image - min_value) / (max_value + 0.00001)
    cam_map = cv2.resize(np.array(cam_map, 'float32'), (shape_image[0], shape_image[1]))  # must in float to resize
    weight_map = cam_map / (np.max(cam_map) + 0.00001)
    weight_map_image = 1 - weight_map
    return_image = np.zeros((shape_image[0], shape_image[1] * 2, 3), 'float32')
    if len(shape_image) == 2:
        return_image[:, 0: shape_image[1], 0] = data_image
        return_image[:, 0: shape_image[1], 1] = data_image
        return_image[:, 0: shape_image[1], 2] = data_image
    else:
        return_image[:, 0: shape_image[1], :] = data_image

    return_image[:, shape_image[1]::, 0] = \
        weight_map_image * return_image[:, 0: shape_image[1], 0] + weight_map * heat_map[:, :, 0]
    return_image[:, shape_image[1]::, 1] = \
        weight_map_image * return_image[:, 0: shape_image[1], 1] + weight_map * heat_map[:, :, 1]
    return_image[:, shape_image[1]::, 2] = \
        weight_map_image * return_image[:, 0: shape_image[1], 2] + weight_map * heat_map[:, :, 2]
    return return_image


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


def merge_image_with_mask_3d(image_array, mask_array, save_path=None, high_resolution=True,
                             clip_window=None, z_to_plot=None, show=True):
    """

    :param show:
    :param z_to_plot:
    :param clip_window:
    :param high_resolution:
    :param save_path:
    :param image_array:
    :param mask_array: binary
    :return:
    """
    if z_to_plot is None:
        z_to_plot = int(np.median(list(np.where(mask_array > 0.5)[2])))
    if clip_window is not None:
        image = np.clip(image_array[:, :, z_to_plot], clip_window[0], clip_window[1])
    else:
        image = image_array[:, :, z_to_plot]
    return merge_image_with_mask(cast_to_0_1(image), mask_array[:, :, z_to_plot],
                                 save_path=save_path, high_resolution=high_resolution, show=show)


def show_point_cloud_3d(input_data, data_type='array'):
    """
    visualize the point cloud
    :param input_data: binary numpy array in 3d or list of locations list in [(x, y, z), ...]
    :param data_type: 'array' or 'list'
    :return: None
    """
    if type(input_data) == list:
        data_type = "list"
    assert data_type in ['array', 'list']
    if data_type == 'array':
        assert len(np.shape(input_data)) == 3
        location_array = np.where(input_data > 0.5)

    else:
        assert len(input_data) > 0
        assert len(input_data[0]) == 3
        location_array = list(zip(*input_data))

    x = location_array[0]
    y = location_array[1]
    z = location_array[2]
    ax = plt.gca(projection="3d")
    ax.scatter(x, y, z, s=1)
    plt.show()
    plt.close()


def get_location_list(numpy_location_array):
    """

    :param numpy_location_array: return for numpy.where
    :return: list of locations for each voxels [(x, y, z, ...), ...]
    """
    return list(zip(*numpy_location_array))


def get_location_array(location_list, dtype='int32'):
    """

    :param dtype:
    :param location_list: list of locations for each voxels [(x, y, z, ...), ...]
    :return: numpy array in shape [len(source_arrays[0]), len(source_arrays)]
    """
    assert len(location_list) > 0
    return_list = []
    temp_array = np.array(list(zip(*location_list)))
    dimension = len(location_list[0])
    for index in range(dimension):
        return_list.append(np.array(temp_array[index], dtype))
    return tuple(return_list)


if __name__ == '__main__':
    test_array = np.array([2,4,1,4,5,3,5,3,5,3,])
    test_array.sort()
    print(test_array[(len(test_array) - 5):])
    exit()
    test_list = [1, 3, 4, 5, 5, 5, 6, 7, 8, 11]
    print(find_index_for_value_in_sorted_object(test_list, 5))

    exit()

    print(time.time())
    exit()
    test_list = pickle_load_object('/home/zhoul0a/Desktop/test_list.pickle')
    print(test_list)
    test_list = [1234444, 222]
    pickle_save_object('/home/zhoul0a/Desktop/test_list.pickle', test_list)
    test_list = pickle_load_object('/home/zhoul0a/Desktop/test_list.pickle')
    print(test_list)
    exit()
    fn_list = os.listdir('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/')
    dict_name = 'stage_two_last_cnn_version4'
    for fn in fn_list:
        print(fn)
        data = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/' + dict_name + '/' + fn[
                                                                                                             :-4] + '_sample_Z.npy')[:, :, 1]
        data = np.swapaxes(data, 0, 1)
        heat_map_cam = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/' + dict_name + '/' + fn[
                                                                                                             :-4] + '_heatmap_Z.npy')
        heat_map_cam = np.swapaxes(heat_map_cam, 0, 1)
        final_image = merge_with_heat_map(data, heat_map_cam)

        image_save(final_image, '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/picture_cam/' + dict_name + '/' + fn[:-4] + '.png', high_resolution=True)


def set_visible_gpu(visible_device):
    if visible_device is None:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device


def set_visible_device(device_ids):  # like '0, 1'
    if device_ids is None:
        return None
    assert type(device_ids) is str
    return set_visible_gpu(device_ids)


def split_list_by_ord_sum(string_list, fold=(0, 1), sort=True):
    split_list = []
    if sort:
        string_list.sort()
    for string in string_list:
        assert type(string) is str
        ord_sum = 0
        for char in string:
            ord_sum += ord(char)
        if ord_sum % fold[1] == fold[0]:
            split_list.append(string)
    if fold[0] < 0:
        split_list.reverse()
    return split_list


def set_stable_hash():
    import sys
    if not os.environ.get('PYTHONHASHSEED'):
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, ['python3'] + sys.argv)


def stable_string_hash(my_string):
    return int(hashlib.sha512(my_string.encode('utf-8')).hexdigest(), 16)


def get_ord_sum(string):
    ord_sum = 0
    for char in string:
        ord_sum += ord(char)
    return ord_sum
