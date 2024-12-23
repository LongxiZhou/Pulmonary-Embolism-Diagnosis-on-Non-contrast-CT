import Tool_Functions.Functions as Functions
import os
import bintrees
import numpy as np
import pydicom
import SimpleITK as sitk


def stack_dcm_with_instance_id(dic, show=True):

    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    if show:
        print("number_dcm_files:", num_slices)
    first_slice = Functions.load_dicom(os.path.join(dic, dcm_file_names[0]), load_wit_sitk=False)[0]

    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    slice_id_list = []
    array_3d = np.zeros([rows, columns, num_slices], 'float32')

    value_id_list = []

    for file in dcm_file_names:
        data_array, slice_id = Functions.load_dicom(os.path.join(dic, file), load_wit_sitk=False)

        value_id_list.append((data_array, slice_id))

        slice_id_list.append(slice_id)
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)

    max_slice_id = np.max(slice_id_list)
    min_slice_id = np.min(slice_id_list)
    assert max_slice_id - min_slice_id + 1 == len(slice_id_list)

    for values in value_id_list:
        array_3d[:, :, values[1] - min_slice_id] = values[0]

    if show:
        print('the array has shape:', np.shape(array_3d))
        Functions.array_stat(array_3d)
        print('stack complete!')
    return array_3d


def get_resolution_from_dcm(dcm_dict):
    dcm_file_names = os.listdir(dcm_dict)
    for file_name in dcm_file_names:
        content = pydicom.read_file(os.path.join(dcm_dict, file_name))
        if hasattr(content, 'PixelSpacing') and hasattr(content, 'SliceThickness'):
            resolution = content.PixelSpacing
            resolution.append(content.SliceThickness)
            return resolution

    raise ValueError("resolution cannot be resolved.")


def stack_dcm_files_and_get_resolution_simplest(dic, show=True):
    return stack_dcm_with_instance_id(dic, show=show), get_resolution_from_dcm(dic)


def stack_dcm_files_simplest_v2(dic, show=True):

    dcm_file_names = os.listdir(dic)

    num_slices = len(dcm_file_names)

    if show:

        print("number_dcm_files:", num_slices)

    first_slice = Functions.load_dicom(os.path.join(dic, dcm_file_names[0]))[0]

    first_content = pydicom.read_file(os.path.join(dic, dcm_file_names[0]))

    resolutions = first_content.PixelSpacing

    try:
        resolutions.append(first_content.SliceThickness)
    except:
        print("slice thickness cannot resolve, set to 1")
        resolutions.append(1)

    if show:

        print('the resolution for x, y, z in mm:', resolutions)

    rows, columns = first_slice.shape

    reader = sitk.ImageSeriesReader()

    dcm_series = reader.GetGDCMSeriesFileNames(dic)

    reader.SetFileNames(dcm_series)

    img = reader.Execute()

    img_array = sitk.GetArrayFromImage(img)  # z y x

    img_array = np.swapaxes(img_array, 0, 2)

    img_array = np.swapaxes(img_array, 0, 1)

    if show:

        print('the array corresponds to a volume of:',

              rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])

        Functions.array_stat(img_array)

        print('stack complete!')

    return img_array, resolutions


def stack_dcm_files(dic, show=True, wc_ww=(-600, 1600), use_default=True):
    # the dictionary like '/home/zhoul0a/CT_slices_for_patient_alice/'
    # wc_ww should be a tuple like (-600, 1600) if you want to to assign wc_ww.
    # return a 3D np array with shape [Rows, Columns, Num_Slices], and the resolution of each axis_list
    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    first_slice = Functions.load_dicom(os.path.join(dic, dcm_file_names[0]))[0]
    try:
        wc, ww = Functions.wc_ww(os.path.join(dic, dcm_file_names[0]))
    except:
        print("no ww and wc, use default")
        wc, ww = wc_ww
    if (wc < -800 or wc > -400 or ww > 1800 or ww < 1400) and use_default:
        print("the original wc, ww is:", wc, ww, "which is strange, we use default.")
    if wc_ww is not None:
        wc, ww = wc_ww
    print('the window center and window width are:')
    print("\n### ", wc, ",", ww, '###\n')
    first_content = pydicom.read_file(os.path.join(dic, dcm_file_names[0]))
    resolutions = first_content.PixelSpacing
    try:
        resolutions.append(first_content.SliceThickness)
    except:
        print("slice thickness cannot resolve, set to 1")
        resolutions.append(1)
    if show:
        print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    array_3d = np.zeros([rows, columns, num_slices], 'int32')
    slice_id_list = []
    for file in dcm_file_names:
        data_array, slice_id = Functions.load_dicom(os.path.join(dic, file))
        slice_id_list.append(slice_id)
        slice_id -= 1
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)
        array_3d[:, :, num_slices - slice_id - 1] = data_array
    if show:
        print('the array corresponds to a volume of:', rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])
    Functions.array_stat(array_3d)
    array_3d -= wc
    array_3d = array_3d / ww  # cast the lung signal into -0.5 to 0.5
    print('stack complete!')
    slice_id_list.sort()
    print(slice_id_list)

    return array_3d, resolutions


def get_ct_array(patient_id, show=False):
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    array_list = []
    for time in time_points:
        array, _ = stack_dcm_files(top_dic + time + '/Data/raw_data/', show)
        array_list.append(array)
    return array_list, time_points


def get_info(patient_id, show=False):
    if show:
        print('get information for patient:', patient_id)
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    if show:
        print('we have these time points:', time_points)
    resolutions_list = []  # the elements are [x1, y1, z1], [x2, y2, z2], ...
    shape_list = []
    for time in time_points:
        data_dict = top_dic + time + '/Data/raw_data/'
        dcm_list = os.listdir(data_dict)
        num_slices = len(dcm_list)
        first_slice = Functions.load_dicom(data_dict + dcm_list[0])[0]
        rows, columns = first_slice.shape
        first_content = pydicom.read_file(data_dict + dcm_list[0])
        resolutions = first_content.PixelSpacing
        try:
            resolutions.append(first_content.SliceThickness)
        except:
            print("slice thickness cannot resolve, set to 1")
            resolutions.append(1)
        shape = [rows, columns, num_slices]
        shape_list.append(shape)
        resolutions_list.append(resolutions)
        wc, ww = Functions.wc_ww(data_dict + dcm_list[0])
        if show:
            print('time point', time, 'has shape', shape, ', resolution', resolutions, ', wc ww', wc, ww)
    if show:
        print('\n')
    return time_points, shape_list, resolutions_list


if __name__ == '__main__':
    exit()
