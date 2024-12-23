import numpy as np
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import \
    reconstruct_semantic_from_sample_sequence
import os
import Tool_Functions.Functions as Functions


def analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, show=True):

    artery_volume = np.sum(artery_mask) + 0.001
    vein_volume = np.sum(vein_mask) + 0.001

    clot_in_artery = predict_clot_mask * artery_mask
    clot_in_vein = predict_clot_mask * vein_mask

    ratio_clot_artery = np.sum(clot_in_artery) / artery_volume
    total_artery_clot_volume = ratio_clot_artery * artery_volume * (334 / 512 * 334 / 512)
    ratio_clot_vein = np.sum(clot_in_vein) / vein_volume
    total_vein_fp_volume = ratio_clot_vein * vein_volume * (334 / 512 * 334 / 512)
    if show:
        print("a-v clot ratio:", ratio_clot_artery / ratio_clot_vein, "   artery clot ratio:", ratio_clot_artery,
              "   artery clot volume in mm^3", total_artery_clot_volume, "   vein fp ratio:", ratio_clot_vein,
              "   vein fp volume in mm^3", total_vein_fp_volume)

    return ratio_clot_artery / ratio_clot_vein, ratio_clot_artery, total_artery_clot_volume, \
        ratio_clot_vein, total_vein_fp_volume


def get_metrics_case(patient_name, high_resolution=False):

    if len(patient_name) > 7:
        if patient_name[-7:] == '.pickle':
            patient_name = patient_name[:-7]

    if high_resolution:
        absolute_cube_length = (4, 4, 5)
        top_dict_sample_sequence = '/data_disk/RSNA-PE_dataset/sample_sequence/' \
                                   'pe_v3_inference_result/(hi-reso, denoise, sim-non, big-roi): 1000'
    else:
        absolute_cube_length = (7, 7, 10)
        top_dict_sample_sequence = '/data_disk/RSNA-PE_dataset/sample_sequence/' \
                                   'pe_v3_inference_result/(hi-reso, denoise, sim-non, big-roi): 0000'

    sample_sequence_path = os.path.join(top_dict_sample_sequence, patient_name + '.pickle')
    sample_sequence = Functions.pickle_load_object(sample_sequence_path)

    artery_dict = '/data_disk/RSNA-PE_dataset/semantics/artery_mask'
    vein_dict = '/data_disk/RSNA-PE_dataset/semantics/vein_mask'

    artery_mask = np.load(os.path.join(artery_dict, patient_name + '.npz'))['array']
    vein_mask = np.load(os.path.join(vein_dict, patient_name + '.npz'))['array']

    depth_dict = '/data_disk/RSNA-PE_dataset/depth_and_center-line/depth_array'
    depth_vessel = np.load(os.path.join(depth_dict, patient_name + '.npz'))['array']

    strict_mask = np.array(depth_vessel > 2.5, 'float32')

    artery_strict = strict_mask * artery_mask
    vein_strict = strict_mask * vein_mask

    clot_prob = reconstruct_semantic_from_sample_sequence(
        sample_sequence, absolute_cube_length, key='clot_prob_mask', background=0)

    av_ratio, artery_clot_ratio, total_artery_clot_volume, ratio_clot_vein, total_vein_fp_volume = \
        analysis_clot_in_av(clot_prob, artery_mask, vein_mask)

    av_ratio_strict, artery_clot_ratio_strict, total_artery_clot_volume_strict, ratio_clot_vein_strict, \
        total_vein_fp_volume_strict = analysis_clot_in_av(clot_prob, artery_strict, vein_strict)

    metrics = {'avr': av_ratio, 'acr': artery_clot_ratio, 'tcv': total_artery_clot_volume,
               'vcr': ratio_clot_vein, 'tvv': total_vein_fp_volume,
               'avr_strict': av_ratio_strict, 'acr_strict': artery_clot_ratio_strict,
               'tcv_strict': total_artery_clot_volume_strict,
               'vcr_strict': ratio_clot_vein_strict, 'tvv_strict': total_vein_fp_volume_strict}

    return metrics


def get_metrics_dataset(high_resolution=False, fold=(0, 1)):
    if high_resolution:
        save_dict = '/data_disk/RSNA-PE_dataset/pickle_objects/av_metrics_high_reso'
        sample_dict = '/data_disk/RSNA-PE_dataset/sample_sequence/pe_v3_inference_result/' \
                      '(hi-reso, denoise, sim-non, big-roi): 1000'
    else:
        save_dict = '/data_disk/RSNA-PE_dataset/pickle_objects/av_metrics_low_reso'
        sample_dict = '/data_disk/RSNA-PE_dataset/sample_sequence/pe_v3_inference_result/' \
                      '(hi-reso, denoise, sim-non, big-roi): 0000'

    fn_list = os.listdir(sample_dict)
    fn_list = Functions.split_list_by_ord_sum(fn_list, fold=fold)

    count = 0
    for fn in fn_list:
        print("processing", fn, count, '/', len(fn_list))
        save_path = os.path.join(save_dict, fn)
        if os.path.exists(save_path):
            print("processed")
            count += 1
            continue
        metrics = get_metrics_case(fn, high_resolution)
        print("metrics:", metrics)
        Functions.pickle_save_object(save_path, metrics)
        count += 1


if __name__ == '__main__':
    get_metrics_dataset(fold=(0, 3))
