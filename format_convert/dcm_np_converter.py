"""
provide these functions. numpy is the standard format to process
dcm -> npy unrescaled
dcm -> npy signal rescaled
dcm -> npy spatial rescaled
dcm -> npy spatial and signal rescaled
mha -> npy
npy -> mha
npy spatial rescaled -> npy spatial unrescaled  (convert standard shape and resolution to original ones)
"""
import format_convert.read_in_CT as read_in_CT
from medpy import io
import SimpleITK as si
import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.spatial_normalize as spatial_normalize
import pydicom
import os


def dcm_to_unrescaled(dcm_dict, save_path=None, show=True, return_resolution=False):
    """
    just stack dcm files together
    :param return_resolution:
    :param show:
    :param dcm_dict:
    :param save_path: the save path for stacked array
    :return: the stacked array in float32
    """
    array_stacked, resolution = read_in_CT.stack_dcm_files_simplest_v2(dcm_dict, show=show)
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, array_stacked)
    if return_resolution:
        return array_stacked, resolution
    return array_stacked


def dcm_to_signal_rescaled(dcm_dict, wc_ww=None, save_path=None, show=True):
    unrescaled_array = dcm_to_unrescaled(dcm_dict, save_path=None, show=show)
    if wc_ww is None:
        dcm_file_names = os.listdir(dcm_dict)
        wc, ww = Functions.wc_ww(os.path.join(dcm_dict, dcm_file_names[0]))
        if show:
            print("no wc_ww given, using default. wc:", wc, " ww:", ww)
    else:
        wc, ww = wc_ww
        if show:
            print("given wc_ww wc:", wc, " ww:", ww)
    signal_rescaled = (unrescaled_array - wc) / ww  # cast the wc_ww into -0.5 to 0.5
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, signal_rescaled)
    return signal_rescaled


def dcm_to_spatial_rescaled(dcm_dict, target_resolution=(334 / 512, 334 / 512, 1), target_shape=(512, 512, 512),
                            save_path=None, show=True, tissue='lung', return_resolution=False,
                            return_original_shape=False):
    unrescaled_array, resolution = dcm_to_unrescaled(dcm_dict, save_path=None, show=show, return_resolution=True)
    if tissue == 'lung':
        assert target_resolution == (334 / 512, 334 / 512, 1)
        assert target_shape == (512, 512, 512)
    spatial_rescaled = spatial_normalize.rescale_to_standard(unrescaled_array, resolution, target_resolution,
                                                             target_shape, tissue=tissue)
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, spatial_rescaled)
    if return_resolution and return_original_shape:
        return spatial_rescaled, resolution, np.shape(unrescaled_array)
    if return_resolution:
        return spatial_rescaled, resolution
    if return_original_shape:
        return spatial_rescaled, np.shape(unrescaled_array)
    return spatial_rescaled


def dcm_to_spatial_signal_rescaled(dcm_dict, wc_ww=(-600, 1600), target_resolution=(334 / 512, 334 / 512, 1),
                                   target_shape=(512, 512, 512), tissue='chest', save_path=None, show=True,
                                   return_resolution=False):
    """

    :param dcm_dict:
    :param wc_ww:
    :param target_resolution:
    :param target_shape:
    :param tissue:
    :param save_path:
    :param show:
    :param return_resolution: here the resolution is the resolution for the original dcm files
    :return:
    """
    if tissue == 'chest':
        assert target_resolution == (334 / 512, 334 / 512, 1)
        assert target_shape == (512, 512, 512)
    if wc_ww is None:
        dcm_file_names = os.listdir(dcm_dict)
        wc, ww = Functions.wc_ww(os.path.join(dcm_dict, dcm_file_names[0]))
        if show:
            print("no wc_ww given, using default. wc:", wc, " ww:", ww)
    else:
        wc, ww = wc_ww
        if show:
            print("given wc_ww wc:", wc, " ww:", ww)
    spatial_rescaled, resolution = dcm_to_spatial_rescaled(dcm_dict, target_resolution, target_shape, None, show,
                                                           tissue, return_resolution=True)
    spatial_signal_rescaled = (spatial_rescaled - wc) / ww
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, spatial_signal_rescaled)
    if return_resolution:
        return spatial_signal_rescaled, resolution
    return spatial_signal_rescaled


def to_rescaled_ct_for_chest_ct(dcm_dict, return_resolution=False, show=True):
    """

    :param dcm_dict:
    :param return_resolution: here the resolution is the original resolution for dcm files
    :param show
    :return: rescaled_ct, resolution of the original dcm files in (x, y, z) (optional)
    """
    return dcm_to_spatial_signal_rescaled(dcm_dict, (-600, 1600), tissue='chest', save_path=None, show=show,
                                          return_resolution=return_resolution)


def rescaled_pipeline_for_arranged_dataset(raw_data_top_dict, save_top_dict, compress=False, wc_ww=(-600, 1600)):
    # TODO
    """
    dataset should be arranged as raw_data_top_dict/patient-id/time-point/Data/raw_data/.dcm files
    rescaled numpy array will be in saved as save_top_dict/patient-id_time-point.npy(.npz)
    :param wc_ww:
    :param raw_data_top_dict:
    :param save_top_dict:
    :param compress: whether save with npz
    :return: None
    """
    patient_id_list = os.listdir(raw_data_top_dict)
    processed = 0
    for patient in patient_id_list:
        print(len(patient_id_list) - processed, "left")
        scan_top_dict = os.path.join(raw_data_top_dict, patient)
        time_point_list = os.listdir(scan_top_dict)
        for time in time_point_list:
            print("processing patient", patient, "at time", time)
            year, month, date = time.split('-')
            if len(month) == 1:
                month = '0' + month
            if len(date) == 1:
                date = '0' + date
            time_to_save = '2021' + month + date

            if os.path.exists(os.path.join(save_top_dict, patient + '_' + time_to_save + '.npz')):
                print("processed")
                continue
            elif os.path.exists(os.path.join(save_top_dict, patient + '_' + time_to_save + '.npy')):
                print("processed")
                continue
            else:
                dcm_top_dict = os.path.join(scan_top_dict, time) # + '/Data/raw_data/'
                rescaled_array = dcm_to_spatial_signal_rescaled(dcm_top_dict, wc_ww=wc_ww, show=False)
                Functions.save_np_array(save_top_dict, patient + '_' + time_to_save, rescaled_array, compress=compress)
        processed += 1
    return None


def read_in_mha(path):
    ar = si.ReadImage(path)
    mask = si.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0.5, 'float32')
    return mask  # (x, y, z)


def save_np_as_mha(np_array, save_dict, file_name):
    # only for binary mask
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if file_name[-4::] == '.mha':
        file_name = file_name[:-4]

    np_array = np.transpose(np_array, (1, 0, 2))
    np_array[np_array < 0.5] = 0
    np_array[np_array >= 0.5] = 1
    np_array = np_array.astype("uint8")
    header = io.Header(spacing=(1, 1, 1))
    print("mha file path:", os.path.join(save_dict, file_name) + '.mha')
    io.save(np_array, os.path.join(save_dict, file_name) + '.mha', hdr=header, use_compression=True)


def get_original_resolution(dcm_dict, tissue='lung'):
    dcm_file_names = os.listdir(dcm_dict)
    num_slices = len(dcm_file_names)
    first_slice = Functions.load_dicom(os.path.join(dcm_dict, dcm_file_names[0]))[0]
    first_content = pydicom.read_file(os.path.join(dcm_dict, dcm_file_names[0]))
    resolution = first_content.PixelSpacing
    try:
        resolution.append(first_content.SliceThickness)
    except:
        print("slice thickness cannot resolve, set to 1")
        resolution.append(1)
    rows, columns = first_slice.shape
    # the original shape should be [rows, columns, num_slices]
    original_shape = (rows, columns, num_slices)
    if tissue == 'lung' and original_shape[2] * resolution[2] > 450:
        resolution[2] = 450 / original_shape[2]
    return resolution


def undo_spatial_rescale(dcm_dict, spatial_rescaled_array, resolution_rescaled=(334 / 512, 334 / 512, 1),
                         tissue='lung'):
    """
    align to the original dcm files, e.g. mask[:, :, slice_id] is for dcm file of slice_id
    :param dcm_dict:
    :param spatial_rescaled_array: the prediction is on the rescaled array
    :param resolution_rescaled: the resolution of the standard space
    :param tissue:
    :return: array that undo the spatial rescale
    """
    if tissue == 'lung':
        assert resolution_rescaled == (334 / 512, 334 / 512, 1)
    dcm_file_names = os.listdir(dcm_dict)
    num_slices = len(dcm_file_names)
    first_slice = Functions.load_dicom(os.path.join(dcm_dict, dcm_file_names[0]))[0]
    first_content = pydicom.read_file(os.path.join(dcm_dict, dcm_file_names[0]))
    resolution = first_content.PixelSpacing
    try:
        resolution.append(first_content.SliceThickness)
    except:
        print("slice thickness cannot resolve, set to 1")
        resolution.append(1)
    rows, columns = first_slice.shape
    # the original shape should be [rows, columns, num_slices]
    original_shape = (rows, columns, num_slices)
    if tissue == 'lung' and original_shape[2] * resolution[2] > 450:
        resolution[2] = 450 / original_shape[2]
    return spatial_normalize.rescale_to_original(spatial_rescaled_array, resolution_rescaled, resolution,
                                                 original_shape)


def normalize_gt_array(dcm_dict, raw_gt_array, tissue='lung', target_resolution=(334/512, 334/512, 1),
                       target_shape=(512, 512, 512)):
    if tissue == 'lung':
        assert target_shape == (512, 512, 512)
        assert target_resolution == (334/512, 334/512, 1)
    resolution_raw = get_original_resolution(dcm_dict, tissue=tissue)
    min_gt, max_gt = np.min(raw_gt_array), np.max(raw_gt_array)
    print("min_gt:", min_gt, "max_gt:", max_gt)
    if max_gt - min_gt == 0:
        print("max and min is the same: max", max_gt, "min:", min_gt)
    if max_gt > 1:
        print("max gt is greater than 1")
        assert max_gt - min_gt > 0
    if not min_gt == 0:
        print("min gt not equals to zero")

    return spatial_normalize.rescale_to_standard(raw_gt_array, resolution_raw, target_resolution, target_shape)


if __name__ == '__main__':
    data_set_dict = '/home/zhoul0a/Desktop/pulmonary nodules/data_temp/dcm_and_gt/'
    patient_list = os.listdir(data_set_dict)
    for patient in patient_list:
        print(patient)
        save_name = patient + '_' + '2020-05-01'
        if os.path.exists('/home/zhoul0a/Desktop/pulmonary nodules/data_temp/rescaled_ct/' + save_name + '.npy'):
            print("processed")
            continue
        data_dict = data_set_dict + patient + '/2020-05-01/' + 'Data/'
        ct_rescaled, ct_resolution = dcm_to_spatial_signal_rescaled(data_dict + 'raw_data/', return_resolution=True)
        gt_raw = Functions.read_in_mha(data_dict + 'ground_truth/PN.mha')
        gt_rescaled = spatial_normalize.rescale_to_standard(gt_raw, ct_resolution)
        gt_rescaled = np.array(gt_rescaled > 0.0001, 'float32')

        Functions.save_np_array('/home/zhoul0a/Desktop/pulmonary nodules/data_temp/rescaled_ct/', save_name, ct_rescaled)
        Functions.save_np_array('/home/zhoul0a/Desktop/pulmonary nodules/data_temp/rescaled_gt/', save_name, gt_rescaled,
                                compress=True)
