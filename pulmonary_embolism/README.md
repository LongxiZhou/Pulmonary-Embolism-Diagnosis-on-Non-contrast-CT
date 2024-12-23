# The 3D transformer with CNN to convert cube to embedding vector and small cubes

Pre-train with the healthy CT scans, the dataset preparation follows these steps:
1. Form rescaled ct and segment semantics, put them in the following files:
rescaled_ct = np.load(top_dict_normal + file_name)
lung_mask = np.load(top_dict_semantic + 'lung_mask/' + ...
artery_mask = np.load(top_dict_semantic + 'artery_mask/' + ...
vein_mask = np.load(top_dict_semantic + 'vein_mask/' + ...
blood_robust_mask = np.load(top_dict_semantic + 'blood_mask/' + ...
airway_mask = np.load(top_dict_semantic + 'airway_mask/' + ...

2. Check the data quality by .transformer_for_3D.dataset_check.py
Ensure: the segmentation quality for airways and blood vessels should be acceptable,
because the loss function related with the semantic volume, if too less semantic volume,
like no airway detected, the loss will be NA.

3. Modify function "get_penalty_array" in .transformer_for_3D.convert_ct_to_sliced_sequence.py
Like for CT with lesions, you may add loss for lesion semantic

4. See function "pipeline_process()" in .transformer_for_3D.convert_ct_to_sliced_sequence.py
change the directory in this function, run it, get dataset for training.

