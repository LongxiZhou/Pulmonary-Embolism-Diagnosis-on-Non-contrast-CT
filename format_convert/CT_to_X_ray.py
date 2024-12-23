import numpy as np
import Tool_Functions.Functions as Functions

ct = np.load('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/rescaled_ct/patient-id-20331817.npz')['array']

new_array = np.zeros([512, 512])

for i in range(0, 512):
    new_array = new_array + ct[i, :, :]
new_array = Functions.rotate_and_scale_image(new_array, 90)
Functions.image_show(new_array, gray=True)

exit()
ct = np.clip(ct + 0.5, 0, 1)
blood_mask = np.load('/home/zhoul0a/Desktop/pulmonary_embolism/refine_dataset/basic_semantics/blood_mask/patient-id-S45340.npz')['array']

for i in range(50, 500, 20):
    Functions.merge_image_with_mask(ct[:, :, i], blood_mask[:, :, i])
