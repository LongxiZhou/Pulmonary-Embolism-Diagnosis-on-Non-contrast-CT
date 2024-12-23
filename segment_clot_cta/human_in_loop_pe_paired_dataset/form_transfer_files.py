import pe_dataset_management.basic_functions as basic_functions
from format_convert.dcm_np_converter_new import undo_spatial_rescale
import os
import Tool_Functions.Functions as Functions
import numpy as np


target_dict = '/data_disk/CTA-CT_paired-dataset/transfer/human_in_loop/PE_pre_annotated'
# target_dict/scan_name/  CTA_folder, clot_mask_predict.npz

scan_name_list_no_gt = basic_functions.get_file_name_do_not_have_clot_gt()
print(len(scan_name_list_no_gt))

fold = (0, 3)
scan_name_list_no_gt = scan_name_list_no_gt[fold[0]::fold[1]]
processed = 0

for scan_name in scan_name_list_no_gt:
    print("processing", scan_name, processed, '/', len(scan_name_list_no_gt))
    save_dict = os.path.join(target_dict, scan_name)
    if os.path.exists(os.path.join(save_dict, scan_name + '_clot_predict.npz')):
        print("processed")
        processed += 1
        continue

    source_dict_cta = basic_functions.find_original_dcm_folders(scan_name)[0]
    Functions.copy_file_or_dir(source_dict_cta, os.path.join(save_dict, 'CTA'))

    top_dict_clot_predict = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)[0]
    clot_path = os.path.join(top_dict_clot_predict, 'semantics', 'blood_clot', scan_name + '.npz')

    clot_array = np.load(clot_path)['array']

    clot_array_original = undo_spatial_rescale(clot_array, original_dcm_dict=source_dict_cta)
    clot_array_original = np.array(clot_array_original > 0.5, 'float32')

    Functions.save_np_array(save_dict, scan_name + '_clot_predict.npz', clot_array_original, compress=True)
    processed += 1
