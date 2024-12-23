import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
from chest_ct_database.feature_manager.add_simulated_non_contrast_from_cta import add_simulated_non_contrast
import os
from pe_dataset_management.basic_functions import get_dataset_relative_path


def pipeline_convert_cta_to_ct_paired(
        top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_CTA', current_fold=(0, 1)):
    name_list_dataset = get_dataset_relative_path()
    for dataset in name_list_dataset:
        top_dict = os.path.join(top_dict_database, dataset)
        if not os.path.exists(top_dict):
            continue
        print(top_dict)
        add_simulated_non_contrast(os.path.join(top_dict, 'rescaled_ct-denoise'),
                                   os.path.join(top_dict, 'semantics'),
                                   os.path.join(top_dict, 'simulated_non_contrast'),
                                   fold=current_fold)


# this can remove outlier in heart and vessel regions
def pipeline_add_simulated_contrast_for_non(
        top_dict_database='/data_disk/CTA-CT_paired-dataset/dataset_non_contrast', current_fold=(0, 1)):
    name_list_dataset = get_dataset_relative_path()
    for dataset in name_list_dataset:
        top_dict = os.path.join(top_dict_database, dataset)
        if not os.path.exists(top_dict):
            continue
        print(top_dict)
        add_simulated_non_contrast(os.path.join(top_dict, 'rescaled_ct-denoise'),
                                   os.path.join(top_dict, 'semantics'),
                                   os.path.join(top_dict, 'simulated_non_contrast'),
                                   fold=current_fold)


def pipeline_convert_cta_to_ct(top_dict_database='/data_disk/Altolia_share/PENet_dataset', current_fold=(0, 1)):

    top_dict = top_dict_database

    print(top_dict)
    add_simulated_non_contrast(os.path.join(top_dict, 'rescaled_ct-denoise'),
                               os.path.join(top_dict, 'semantics'),
                               os.path.join(top_dict, 'simulated_non_contrast'),
                               fold=current_fold)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    pipeline_convert_cta_to_ct_paired(current_fold=(0, 8))
