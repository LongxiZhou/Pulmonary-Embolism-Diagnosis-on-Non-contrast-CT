import numpy as np
import os
import Tool_Functions.Functions as Functions


fn_name_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise')

for fn in fn_name_list:
    print("processing:", fn)
    original_vessel = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/blood_mask/'
                              + fn)['array']
    original_artery = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/artery_mask/'
                              + fn)['array']
    original_vein = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/vein_mask/'
                            + fn)['array']
    clot_gt = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_gt/' + fn)['array']

    new_vessel = original_vessel + original_artery + original_vein + clot_gt
    new_vessel = np.clip(new_vessel, 0, 1)

    Functions.save_np_array('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/semantics/'
                            'blood_mask_new/', fn, new_vessel, compress=True)
