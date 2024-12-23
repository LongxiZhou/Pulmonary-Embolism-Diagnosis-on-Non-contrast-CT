import SimpleITK as sitk
import numpy as np
import os


def read_in_mha(path):

    ar = sitk.ReadImage(path)

    mask = sitk.GetArrayFromImage(ar)  # z y x

    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)

    mask = np.array(mask > 0, 'float32')

    return mask  # (x, y, z)


def get_mask_array(patient_id, lung_mask=False):
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    array_list = []
    for time in time_points:
        if not lung_mask:
            print('loading mha')
            array = read_in_mha(top_dic + time + '/Data/ground_truth/LI.mha')
        else:
            print('loading mha')
            array = read_in_mha(top_dic + time + '/Data/ground_truth/右肺(分割).mha')
            array = array + read_in_mha(top_dic + time + '/Data/ground_truth/左肺(分割).mha')
        array_list.append(array)
    return array_list, time_points


def normalize_gt_array(dcm_dict, raw_gt_array, tissue='lung', target_resolution=(334/512, 334/512, 1),
                       target_shape=(512, 512, 512)):
    from format_convert.dcm_np_converter import get_original_resolution
    from format_convert.spatial_normalize import rescale_to_standard
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

    return rescale_to_standard(raw_gt_array, resolution_raw, target_resolution, target_shape)
