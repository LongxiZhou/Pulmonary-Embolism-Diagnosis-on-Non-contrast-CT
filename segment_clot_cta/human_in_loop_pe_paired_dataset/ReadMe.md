for pe_paired_dataset

1. get updated clot segmentation for pe_paired dataset:
run function predict_and_show_dataset in ./longxi_platform/segment_clot_cta/inference/inference_on_pe_paired_dataset.py

2. copy clot_gt to pe_paired_dataset:
run function copy_clot_ground_truth in ./longxi_platform/segment_clot_cta/utlis/operations_pe_paired_dataset.py

3. form folders for transfer