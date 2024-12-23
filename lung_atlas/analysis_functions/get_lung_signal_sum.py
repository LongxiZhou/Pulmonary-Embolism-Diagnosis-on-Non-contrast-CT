"""

we rescale HU by (HU + 1000) / 1000, and fix the spatial resolution, then
when the scan is conducted in a short time,
signal sum in the lung should be a constant
"""


import os
import format_convert.dcm_np_converter_new as converter
import basic_tissue_prediction.predict_rescaled as predictor
import collaborators_package.denoise_chest_ct.denoise_predict as remove_ct_noise
import numpy as np
import visualization.visualize_3d.visualize_stl as stl
import Tool_Functions.Functions as Functions

remove_noise = True
rescaled_ct_a = converter.establish_rescale_chest_ct('/data_disk/CTA-CT_paired-dataset/temp_files/222/S23190/S30')
rescaled_ct_b = converter.establish_rescale_chest_ct('/data_disk/CTA-CT_paired-dataset/temp_files/222/S23190/S50')
if remove_noise:
    rescaled_ct_a = remove_ct_noise.denoise_rescaled_array(rescaled_ct_a)
    rescaled_ct_b = remove_ct_noise.denoise_rescaled_array(rescaled_ct_b)

lung_mask_a = predictor.predict_lung_masks_rescaled_array(rescaled_ct_a)
lung_mask_b = predictor.predict_lung_masks_rescaled_array(rescaled_ct_b)

# stl.visualize_numpy_as_stl(lung_mask_a)
# stl.visualize_numpy_as_stl(lung_mask_b)

rescaled_ct_a = (Functions.change_to_HU(rescaled_ct_a) + 1000) / 1000
rescaled_ct_b = (Functions.change_to_HU(rescaled_ct_b) + 1000) / 1000

threshold = (-100 + 1000) / 1000

pulmonary_region_a = rescaled_ct_a * lung_mask_a
pulmonary_region_a[np.where(pulmonary_region_a > threshold)] = 0  # blood vessel an muscle
pulmonary_region_b = rescaled_ct_b * lung_mask_b
pulmonary_region_b[np.where(pulmonary_region_b > threshold)] = 0  # blood vessel an muscle

sum_a = np.sum(pulmonary_region_a)
sum_b = np.sum(pulmonary_region_b)

print(sum_a, sum_b, sum_a / sum_b)
volume_a = len(np.where(pulmonary_region_a > 0)[0])
volume_b = len(np.where(pulmonary_region_b > 0)[0])
print(volume_a, volume_b, volume_a / volume_b)
