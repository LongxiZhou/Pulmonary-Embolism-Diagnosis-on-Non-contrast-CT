import numpy as np
import os
import Tool_Functions.Functions as Functions


top_dict_dataset = '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/'

fn_name_list = os.listdir(top_dict_dataset + 'rescaled_ct-denoise')

for fn in fn_name_list:
    print("processing:", fn)
    if os.path.exists(top_dict_dataset + 'semantics/'
                      'blood_mask_new/' + fn):
        print("processed")
        continue

    rescaled_ct = np.load(top_dict_dataset + 'rescaled_ct-denoise/'
                          + fn)['array']

    original_vessel = np.load(top_dict_dataset + 'semantics/blood_mask/'
                              + fn)['array']
    original_artery = np.load(top_dict_dataset + 'semantics/artery_mask/'
                              + fn)['array']
    original_vein = np.load(top_dict_dataset + 'semantics/vein_mask/'
                            + fn)['array']

    new_vessel = original_vessel + original_artery + original_vein

    new_vessel = np.clip(new_vessel, 0, 1)

    new_vessel = new_vessel * np.array(rescaled_ct > Functions.change_to_rescaled(-200), 'float32')

    Functions.save_np_array(top_dict_dataset +
                            'semantics/blood_mask_new/', fn, new_vessel, compress=True)
