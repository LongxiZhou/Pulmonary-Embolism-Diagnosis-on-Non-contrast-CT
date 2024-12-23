"""
this is to create rescaled ct from 1 mm dcm files
Provide a
"""

import format_convert.dcm_np_converter as convert
import Tool_Functions.Functions as Functions
import os


def convert_1_mm_dcm_to_rescaled_ct(dict_dcm):
    """
    check whether the dcm data is with enough resolution. The golden standard is the rescaled ct formed by the dcm files
    can achieve satisfactory segmentation for blood vessels for Lung Atlas.
    Here, the function check the dict_dict contains more than 200 files, and check whether the slice_thickness <= 1.5
    :param dict_dcm: the directory for the dcm files
    :return: None for not qualified dcm, or the rescaled ct in shape [512, 512, 512], numpy float32
    """
    file_num = os.listdir(dict_dcm)
    if len(file_num) < 200:
        print("too less dcm files:", len(file_num), " not qualified")
        return None

    rescaled_ct, resolution = convert.dcm_to_spatial_signal_rescaled(dict_dcm, return_resolution=True)
    if resolution[2] > 1.5:
        print("too large thickness, not qualified")
        return None
    return rescaled_ct


def pipeline_rescale_standard_dcm_package(top_dict, save_dict, compress=False):
    """
    Should in top_dict/patient-id/scan-time/Data/raw_data/.dcm
    :param top_dict:
    :param save_dict: save rescaled_ct into top_dict
    :param compress: whether save as npz
    :return: None
    """
    patient_id_list = os.listdir(top_dict)
    for patient in patient_id_list:
        print("processing:", patient)
        scan_time_list = os.listdir(os.path.join(top_dict, patient))
        for scan_time in scan_time_list:

            print('scan time', scan_time, 'for', patient)

            raw_data_dict = os.path.join(top_dict, patient, scan_time, 'Data', 'raw_data')

            rescaled_ct = convert_1_mm_dcm_to_rescaled_ct(raw_data_dict)

            if rescaled_ct is not None:
                Functions.save_np_array(save_dict, patient + '_' + scan_time, rescaled_ct, compress=compress)

    return None


if __name__ == '__main__':
    pipeline_rescale_standard_dcm_package('/home/zhoul0a/Desktop/其它肺炎/3肺部纤维化-59例/哈医大一群力医院-59例',
                                          '/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/rescaled_ct_1mm/fibrosis/')
