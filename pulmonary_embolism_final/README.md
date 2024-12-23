# for each CT scan, it will be converted into a "sample sequence"
sample sequence is form by many patch, which is extracted like convolution though the 3D CT array
we discard a patch if it does not contain blood vessel or only contain very tiny blood vessel

sample sequence is a list, and each element is a dict, recording information of a patch.
keys for the dict:
'ct_data': numpy_array, the patch extracted from CT
'depth_cube': numpy_array, the patch extracted from depth array of the blood vessel segmentation.
              More about depth_array for binary segmentation: for each voxel, outside segmentation is 0,
              inside segmentation is the distance from the nearest surface.
              depth_array = analysis.center_line_and_depth_3D.get_surface_distance(segmentation, strict=False)
'center_location': tuple like (100, 235, 192). the absolute location of the patch
'location_offset': tuple like (-5, 10, 2). the relative offset between this patch and the mass center of blood vessel
'branch_level': float, the average branch level of the vessels inside the patch
'penalty_weight_fp': numpy_array or None, determine voxel penalty for false positive classification.
                     None means penalty_weight_fp is all one
'penalty_weight_fn': numpy_array or None, determine voxel penalty for false negative classification.
                     None means penalty_weight_fn is all one
'clot_gt_mask': numpy_array or None. value from 0 to 1, 1 means this voxel should be classified as clot
'given_vector': numpy_array in shape (L, ) or None. If you want to fine-tune our model with prior-knowledge
                you can assign this value. In this study we set it to None


# For training, each non-contrast CT will be convert to sample, which is a dictionary with keys
"sample_sequence": ,  # list
"center_line_loc_array": ,  # numpy loc array (return of np.where)
"is_PE": ,  # binary or, None (unknown class)
"has_clot_gt": ,  # binary or, None (not PE so clot gt is all zero)
"clot_gt_volume_sum":  # float or, None (no gt or clot gt is all zero)
"registration_quality": str in ('good', 'perfect')
"pe_pair_quality": str in ('good', 'perfect'), 'perfect' means CTA is collected within 1 hour after non-contrast, 'good' means CTA is within 12 hours after non-contrast
"additional_information":  # dict or None


# Training dataset
1. Each batch, half from non-PE non-contrast CT (simulate clot to get gt), half from PE non-contrast CT (with ground truth).
2. sample importance:
1 for non-PE, 1 for PE good register good pair, 1.5 for PE perfect register good pair
2.5 for PE good register perfect pair, 3 for PE perfect register, perfect pair

