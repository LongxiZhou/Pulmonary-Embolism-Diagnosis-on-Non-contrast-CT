import os
import Tool_Functions.Functions as Functions
import numpy as np
import SimpleITK as sitk


def simple_stack_dcm_files(dcm_dict):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dict)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    img_array = sitk.GetArrayFromImage(img)  # z y x
    img_array = np.swapaxes(img_array, 0, 2)
    img_array = np.swapaxes(img_array, 0, 1)
    return img_array


def add_visualization_one_scan(save_dict, interval=2, max_slice=50):
    dict_dcm_cta = os.path.join(save_dict, 'CTA')
    png_dict_check_align = os.path.join(save_dict, 'visualize_clot_predict')
    if os.path.exists(png_dict_check_align):
        print("processed")
        return None

    fn_list = os.listdir(save_dict)
    for fn in fn_list:
        if '.npz' in fn:
            clot_path = os.path.join(save_dict, fn)
            clot_mask_predict = np.load(clot_path)['array']
            cta_array = simple_stack_dcm_files(dict_dcm_cta)
            cta_array = np.clip(cta_array, -800, 400)
            z_list = list(set(np.where(clot_mask_predict > 0.5)[2]))
            z_list.sort()
            z_list = z_list[::interval]
            while len(z_list) > max_slice:
                z_list = z_list[::2]
            for z in z_list:
                image_save_path = os.path.join(png_dict_check_align, str(z) + '.png')
                Functions.merge_image_with_mask(cta_array[:, :, z], clot_mask_predict[:, :, z],
                                                show=False, save_path=image_save_path, dpi=300)
            return None


if __name__ == '__main__':
    top_dict_transfer = '/data_disk/CTA-CT_paired-dataset/transfer/human_in_loop/PE_pre_annotated'
    scan_name_list = os.listdir(top_dict_transfer)
    fold = (0, 5)
    scan_name_list = scan_name_list[fold[0]:: fold[1]]
    processed = 0
    for scan_name in scan_name_list:
        print("processing", scan_name, processed, '/', len(scan_name_list))
        add_visualization_one_scan(os.path.join(top_dict_transfer, scan_name), interval=1)
        processed += 1
