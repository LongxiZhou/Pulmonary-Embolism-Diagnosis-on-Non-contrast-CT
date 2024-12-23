import Tool_Functions.Functions as Functions
import os

# top_dict/patient-id/CTA/dcm_files
# top_dict/patient-id/clot_mask_predict.npz

top_dict_paired_dataset = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/transfer/' \
                          'combine_data/dcm_clot_mask_dataset/'

clot_mask_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/transfer/' \
                     'combine_data/predict_clot_simple_stack'
fn_list = os.listdir(clot_mask_top_dict)

dcm_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/CTA_dcm_files/PE/'

for fn in Functions.iteration_with_time_bar(fn_list):
    print(fn)

    dcm_dict = os.path.join(dcm_top_dict, fn[:-4])
    new_dcm_dict = os.path.join(top_dict_paired_dataset, fn[:-4], 'CTA')
    Functions.copy_file_or_dir(dcm_dict, new_dcm_dict)

    clot_mask_path = os.path.join(clot_mask_top_dict, fn)
    new_clot_mask_path = os.path.join(top_dict_paired_dataset, fn[:-4], 'clot_mask_predict.npz')
    Functions.copy_file_or_dir(clot_mask_path, new_clot_mask_path)
