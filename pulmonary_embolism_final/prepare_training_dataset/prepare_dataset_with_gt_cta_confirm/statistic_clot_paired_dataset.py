"""
Different hospital has different criteria for PE positive. Here we use clot volume to unify the criteria.

{scan_name: {"human_annotation":, "clot_volume":, "clot_volume_artery":, "clot_volume_vein":, "acr":, "vcr":, "avr":, }}
* here artery and vein are high recall mask
* acr, volume clot on artery / artery volume; vcr, volume clot on vein / vein volume; avr = acr / avr
"""
import Tool_Functions.Functions as Functions
import pe_dataset_management.basic_functions as basic_functions
import numpy as np
from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.get_clot_gt_and_penalty import \
    smooth_mask
import os


def get_metric_dict(clot_mask, artery_mask, vein_mask, human_annotation):
    metric_dict = {"human_annotation": human_annotation, "clot_volume": np.sum(clot_mask),
                   "clot_volume_artery": np.sum(clot_mask * artery_mask),
                   "clot_volume_vein": np.sum(clot_mask * vein_mask)}

    artery_volume = np.sum(artery_mask)
    vein_volume = np.sum(vein_mask)

    metric_dict['acr'] = metric_dict["clot_volume_artery"] / artery_volume
    metric_dict['vcr'] = metric_dict["clot_volume_vein"] / vein_volume
    metric_dict['avr'] = metric_dict['acr'] / metric_dict['vcr']

    metric_dict['artery_volume'] = artery_volume
    metric_dict['vein_volume'] = vein_volume

    return metric_dict


def get_metric_for_scan(scan_name):
    top_dict_cta, top_dict_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    scan_name = scan_name + '.npz'

    path_clot_mask = os.path.join(top_dict_cta, 'clot_gt', scan_name)
    human_annotation = True
    if not os.path.exists(path_clot_mask):
        path_clot_mask = os.path.join(top_dict_cta, 'semantics', 'blood_clot', scan_name)
        human_annotation = False
    
    clot_mask = np.load(path_clot_mask)['array']

    # get artery mask
    artery_mask_direct_seg_path = os.path.join(top_dict_cta, 'semantics', 'artery_mask', scan_name)
    artery_mask_seg_on_simulated_non_path = os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'semantics', 'artery_mask', scan_name)
    artery_mask_direct_seg = np.load(artery_mask_direct_seg_path)['array']
    artery_mask_seg_on_simulated_non = np.load(artery_mask_seg_on_simulated_non_path)['array']
    artery_mask = np.clip(artery_mask_direct_seg + artery_mask_seg_on_simulated_non, 0, 1)
    artery_mask = smooth_mask(artery_mask)
    
    # get vein mask
    vein_mask_direct_seg_path = os.path.join(top_dict_cta, 'semantics', 'vein_mask', scan_name)
    vein_mask_seg_on_simulated_non_path = os.path.join(
        top_dict_cta, 'simulated_non_contrast', 'semantics', 'vein_mask', scan_name)
    vein_mask_direct_seg = np.load(vein_mask_direct_seg_path)['array']
    vein_mask_seg_on_simulated_non = np.load(vein_mask_seg_on_simulated_non_path)['array']
    vein_mask = np.clip(vein_mask_direct_seg + vein_mask_seg_on_simulated_non, 0, 1)
    vein_mask = smooth_mask(vein_mask)

    return get_metric_dict(clot_mask, artery_mask, vein_mask, human_annotation)


def process_on_paired_dataset(top_dict_save_pickle='/data_disk/pulmonary_embolism_final/pickle_objects/'
                                                   'scan_name_clot_metric_dict.pickle'):
    if os.path.exists(top_dict_save_pickle):
        scan_name_clot_metric_dict = Functions.pickle_load_object(top_dict_save_pickle)
    else:
        scan_name_clot_metric_dict = {}

    processed_name_set = scan_name_clot_metric_dict.keys()
    scan_name_list_paired_dataset = basic_functions.get_all_scan_name()

    count = 0
    for scan_name in scan_name_list_paired_dataset:
        print("processing:", scan_name, count, '/', len(scan_name_list_paired_dataset))
        if scan_name in processed_name_set:
            print("processed")
            count += 1
            continue
        metric_dict = get_metric_for_scan(scan_name)
        scan_name_clot_metric_dict[scan_name] = metric_dict
        print(metric_dict)
        Functions.pickle_save_object(top_dict_save_pickle, scan_name_clot_metric_dict)
        count += 1


if __name__ == '__main__':
    Functions.set_visible_device('1')
    process_on_paired_dataset()
