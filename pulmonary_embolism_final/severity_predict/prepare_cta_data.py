import os
import numpy as np
import Tool_Functions.Functions as Functions
import pe_dataset_management.basic_functions as basic_functions
from pulmonary_embolism_final.inference.predict_clot_from_sample_sequence import analysis_clot_in_av
from segment_clot_cta.inference.ct_sequence_convert_pe_v3 import reconstruct_rescaled_ct_from_sample_sequence
from segment_clot_cta.inference.inference_pe_v3 import predict_clot_for_sample_sequence
from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided


def process_cta_statistic(scan_name, show_statistics=True, model=None, prediction_only=True):
    # scan_name not with suffix

    save_path = os.path.join(
        '/data_disk/pulmonary_embolism_final/pickle_objects/statistic_clot_cta', scan_name + '.pickle')
    if os.path.exists(save_path):
        print("processed")
        return None

    top_dir_cta, top_dir_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)

    if prediction_only:
        dir_clot_gt = '\n'  # assign an impossible dir
    else:
        dir_clot_gt = os.path.join(top_dir_cta, 'clot_gt')
    if os.path.exists(dir_clot_gt):
        if scan_name + '.npz' in os.listdir(dir_clot_gt):
            path_clot = os.path.join(dir_clot_gt, scan_name + '.npz')
            clot_mask = np.load(path_clot)['array']
        else:
            clot_mask = get_clot_prob(scan_name, model=model)
    else:
        clot_mask = get_clot_prob(scan_name, model=model)

    if clot_mask is None:
        print("sample sequence not prepared for scan:", scan_name)
        return None

    path_artery = os.path.join(top_dir_cta, 'semantics', 'artery_mask', scan_name + '.npz')
    path_vein = os.path.join(top_dir_cta, 'semantics', 'vein_mask', scan_name + '.npz')

    if not os.path.exists(path_artery):
        print("path artery not exist at:", path_artery)
        return None
    if not os.path.exists(path_vein):
        print("path vein not exist at:", path_vein)
        return None

    artery_mask = np.load(path_artery)['array']
    vein_mask = np.load(path_vein)['array']

    statistic = analysis_clot_in_av(clot_mask, artery_mask, vein_mask, show=show_statistics)

    Functions.pickle_save_object(save_path, statistic)

    return None


def get_clot_prob(scan_name, model):
    top_dir_cta, top_dir_non = basic_functions.find_patient_id_dataset_correspondence(scan_name, strip=True)
    relative_dir = 'sample_sequence/pe_v3_inference/(hi-reso, denoise, sim-non, big-roi): 0111'

    sample_sequence_path = os.path.join(top_dir_cta, relative_dir, scan_name + '.pickle')

    if not os.path.exists(sample_sequence_path):
        return None

    sample_sequence = Functions.pickle_load_object(sample_sequence_path)

    sample_sequence_with_clot = predict_clot_for_sample_sequence(sample_sequence, trim=False,
                                                                 high_resolution=False,
                                                                 model=model)
    absolute_cube_length = (7, 7, 10)
    predicted_clot_mask = reconstruct_rescaled_ct_from_sample_sequence(
        sample_sequence_with_clot, absolute_cube_length=absolute_cube_length, key='clot_prob_mask')
    return predicted_clot_mask


def process_all(fold=(0, 1), show_statistics=True):
    scan_name_list = basic_functions.get_all_scan_name()
    scan_name_list = Functions.split_list_by_ord_sum(scan_name_list, fold=fold)

    model = load_saved_model_guided(high_resolution=False, model_path=None)

    processed_count = 0
    for scan_name in scan_name_list:
        print("processing", scan_name, processed_count, '/', len(scan_name_list))
        process_cta_statistic(scan_name, show_statistics=show_statistics, model=model)
        processed_count += 1


if __name__ == '__main__':
    Functions.set_visible_device('1')
    process_all((0, 4))
