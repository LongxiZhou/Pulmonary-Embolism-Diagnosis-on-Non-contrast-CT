import numpy as np
import Tool_Functions.Functions as Functions


sample = Functions.pickle_load_object(
    '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/loop_3_59/pe_v3_long_length_complete_vessel/original_low-resolution/Z104-2022-06-10.pickle')

print(list(sample.keys()))

sample_sequence = sample['sample_sequence']

print(list(sample_sequence[0].keys()))

from segment_clot_cta.inference.ct_sequence_convert_pe_v3 import reconstruct_rescaled_ct_from_sample_sequence

clot_gt = reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, absolute_cube_length=(7, 7, 10), key='clot_array')
ct_array = reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, absolute_cube_length=(7, 7, 10), key='ct_data')

mass_center = Functions.get_mass_center_for_binary(clot_gt)
z = int(mass_center[2])

Functions.merge_image_with_mask(ct_array[:, :, z], clot_gt[:, :, z], show=True)