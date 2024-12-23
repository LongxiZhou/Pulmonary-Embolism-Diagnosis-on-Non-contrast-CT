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


top_dict_patient = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/transfer/combine_data/' \
                'dcm_clot_mask_dataset/Z135'

path_clot_mask = os.path.join(top_dict_patient, 'clot_mask_predict.npz')
clot_mask_predict = np.load(path_clot_mask)['array']

dict_dcm_cta = os.path.join(top_dict_patient, 'CTA')
cta_array = simple_stack_dcm_files(dict_dcm_cta)
cta_array = np.clip(cta_array, -800, 400)


png_dict_check_align = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/' \
                       'transfer/combine_data/check_aligned/Z135'

z_list = list(set(np.where(clot_mask_predict > 0.5)[2]))
z_list.sort()
for z in z_list[::2]:
    image_save_path = os.path.join(png_dict_check_align, str(z) + '.png')
    Functions.merge_image_with_mask(cta_array[:, :, z], clot_mask_predict[:, :, z],
                                    show=False, save_path=image_save_path, dpi=300)
