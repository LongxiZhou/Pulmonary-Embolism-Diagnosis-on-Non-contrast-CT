from format_convert.dcm_np_converter_new import undo_spatial_rescale, simple_stack_dcm_files
import os
import Tool_Functions.Functions as Functions
import numpy as np


source_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_clot_predict/'
target_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/transfer/predict_clot_simple_stack/'

check_image_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/transfer/check_alignment/'

dcm_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/CTA_dcm_files/PE/'

fn_list = os.listdir(source_dict)

for fn in fn_list[1::2]:
    clot_save_path = os.path.join(target_dict, fn)
    if os.path.exists(clot_save_path):
        print("processed.")
        continue

    dcm_dict = os.path.join(dcm_top_dict, fn[:-4])
    simple_stack_array = simple_stack_dcm_files(dcm_dict)
    simple_stack_array = np.clip(simple_stack_array, -800, 400)

    clot_predict_rescaled = np.load(os.path.join(source_dict, fn))['array']
    clot_predict = undo_spatial_rescale(clot_predict_rescaled, dcm_dict)
    clot_predict = np.array(clot_predict > 0.5, 'float32')
    Functions.save_np_array(target_dict, fn, clot_predict, compress=True)

    z_list = list(set(np.where(clot_predict > 0.5)[2]))
    z_list.sort()

    for z in z_list[::2]:
        image_save_path = os.path.join(check_image_top_dict, fn[:-4], str(z) + '.png')
        Functions.merge_image_with_mask(simple_stack_array[:, :, z], clot_predict[:, :, z],
                                        show=False, save_path=image_save_path, dpi=300)

