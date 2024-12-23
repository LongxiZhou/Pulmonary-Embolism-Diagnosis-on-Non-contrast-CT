from pe_dataset_management.basic_functions import find_original_dcm_folders
import format_convert.dcm_np_converter_new as converter
import Tool_Functions.Functions as Functions
import os

new_gt_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/temp_files/mha'
save_rescaled_gt_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_gt_waiting_check'

fn_list = os.listdir(new_gt_dict)
new_gt_set = set()
for fn in fn_list:
    new_gt_set.add(fn[:-4])

new_gt_set = list(new_gt_set)
new_gt_set.sort()
processed_count = 0
for fn in new_gt_set:
    print("processing", fn)
    save_path = os.path.join(save_rescaled_gt_dict, fn + '.npz')
    if os.path.exists(save_path):
        print("processed")
        processed_count += 1
        continue
    dict_dcm_cta, dict_dcm_non_contrast = find_original_dcm_folders(scan_name=fn)
    mha_path = os.path.join(new_gt_dict, fn + '.mha')
    rescaled_gt = converter.establish_rescaled_mask(mha_path, source_dcm_dict=dict_dcm_cta, cast_to_binary=False)

    Functions.save_np_array(save_rescaled_gt_dict, fn + '.npz', rescaled_gt, compress=True)
    processed_count += 1

