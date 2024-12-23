import os
import pe_dataset_management.basic_functions as basics
import Tool_Functions.Functions as Functions


gt_directory_1 = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_gt'
gt_directory_2 = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_gt'


def get_file_names_with_ground_truth():
    fn_set = set()
    fn_set = fn_set | set(os.listdir(gt_directory_1)) | set(os.listdir(gt_directory_2))

    print(len(fn_set))
    print(fn_set)


def copy_clot_ground_truth(original_gt_dict):
    fn_list = os.listdir(original_gt_dict)

    for fn in fn_list:
        cta_dataset_dict = basics.find_patient_id_dataset_correspondence(scan_name=fn, strip=True)[0]
        source_path = os.path.join(original_gt_dict, fn)
        target_path = os.path.join(cta_dataset_dict, 'clot_gt', fn)

        if os.path.exists(target_path):
            print("processed")
            continue

        Functions.copy_file_or_dir(source_path, target_path)


if __name__ == '__main__':
    copy_clot_ground_truth(gt_directory_2)
