import os
import Tool_Functions.Functions as Functions
import numpy as np
import pulmonary_embolism_v3.prepare_training_dataset.trim_refine_and_remove_bad_scan as trim_sequence_length
from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided, predict_clot_for_sample_sequence
import format_convert.spatial_normalize as spatial_normalize


def get_performance_one_sample(sample_sequence_predicted, show=True):
    tp = 0
    fp = 0
    fn = 0

    for cube_dict in sample_sequence_predicted:
        if not np.shape(cube_dict['clot_array']) == (5, 5, 5):
            cube_dict['clot_array'] = spatial_normalize.rescale_to_new_shape(cube_dict['clot_array'], (5, 5, 5))
        clot_array = np.array(cube_dict['clot_array'] > 0.5, 'float32')  # gt mask
        predicted_clot = np.array(cube_dict['clot_prob_mask'] > 0.5, 'float32')  # predicted mask
        tp += np.sum(clot_array * predicted_clot)
        fp += np.sum((1 - clot_array) * predicted_clot)
        fn += np.sum(clot_array * (1 - predicted_clot))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    dice = 2 * recall * precision / (precision + recall)

    if show:
        print('(dice, recall, precision):', (dice, recall, precision), '(tp, fp, fn):', tp, fp, fn)

    return dice, recall, precision, tp, fp, fn


def get_performance_model_and_dataset(model_path, dataset_dict, trim_length=1500, show_instance=True,
                                      fold=None):
    model_loaded = load_saved_model_guided(model_path=model_path)

    fn_list = os.listdir(dataset_dict)

    dice_list = []
    recall_list = []
    precision_list = []
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for fn in fn_list:
        print(fn)
        if fold is not None:
            if not get_fold_path(fn) in fold:
                continue
        sample = Functions.pickle_load_object(os.path.join(dataset_dict, fn))
        sample_sequence = sample['sample_sequence']
        sample_sequence = trim_sequence_length.reduce_sequence_length(sample_sequence, target_length=trim_length,
                                                                      max_branch=9)
        sample_sequence_with_clot = predict_clot_for_sample_sequence(sample_sequence, trim=False, model=model_loaded)

        dice, recall, precision, tp, fp, fn = get_performance_one_sample(sample_sequence_with_clot, show=show_instance)

        dice_list.append(dice)
        recall_list.append(recall)
        precision_list.append(precision)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    recall_overall = tp_total / (tp_total + fn_total)
    precision_overall = tp_total / (tp_total + fp_total)
    dice_overall = 2 * recall_overall * precision_overall / (precision_overall + recall_overall)

    print("overall dice, recall, precision:", (dice_overall, recall_overall, precision_overall))
    print("tp_total, fp_total, fn_total:", (tp_total, fp_total, fn_total))

    return dice_list, recall_list, precision_list, tp_total, fp_total, fn_total


def get_fold_path(path_or_filename):
    file_name = path_or_filename.split('/')[-1]
    ord_sum = 0
    for char in file_name:
        ord_sum += ord(char)
    return ord_sum % 5


if __name__ == '__main__':

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/' \
                               'sample_sequence/PE_with_gt/loop_3_59/' \
                               'pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/' \
                       'gb_0_dice_0.799_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[0])
    # overall dice, recall, precision: (0.5464286826776313, 0.7214773381590853, 0.43973729068484907)
    # tp_total, fp_total, fn_total: (8771.0, 11175.0, 3386.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/next_82/' \
                               'pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_gt/best_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[1, 2, 3, 4])
    # overall dice, recall, precision: (0.5564252675609322, 0.6664461024273319, 0.47758305350455305)
    # tp_total, fp_total, fn_total: (98704.0, 107970.0, 49401.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/next_82/' \
                               'pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/' \
                       'gb_0_dice_0.799_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[1, 2, 3, 4])
    # overall dice, recall, precision: (0.7914853454446409, 0.8436649674217616, 0.7453842620486419)
    # tp_total, fp_total, fn_total: (124951.0, 42682.0, 23154.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/next_82/' \
                               'pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_gt/best_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[0])
    # overall dice, recall, precision: (0.5903355769462739, 0.7221271321489701, 0.4992247429410805)
    # tp_total, fp_total, fn_total: (24470.0, 24546.0, 9416.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_with_gt/next_82/' \
                               'pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_2/' \
                       'gb_0_dice_0.799_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[0])
    # overall dice, recall, precision: (0.7271158657579156, 0.7461488520332881, 0.709029725182277)
    # tp_total, fp_total, fn_total: (25284.0, 10376.0, 8602.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_gt/best_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[1, 2, 3, 4])
    # overall dice, recall, precision: (0.7014298501296734, 0.9519686020433591, 0.5552890730041062)
    # tp_total, fp_total, fn_total: (106967.0, 85666.0, 5397.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_gt/best_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[0, ])
    # overall dice, recall, precision: (0.6007452258965998, 0.6904957350369392, 0.5316424390646038)
    # tp_total, fp_total, fn_total: (19347.0, 17044.0, 8672.0)
    exit()

    gt_sample_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/'
                                'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution')
    for path_sample in gt_sample_list:
        if get_fold_path(path_sample) == 0:
            print(path_sample.split('/')[-1])
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.830_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000,
                                      show_instance=False, fold=[0, ])
    # overall dice, recall, precision: (0.37715826940508035, 0.3044362753845605, 0.49552689671197864)
    # tp_total, fp_total, fn_total: (8530.0, 8684.0, 19489.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only_temp/' \
                       'gb_0_dice_0.832_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.2570693514795799, 0.16442161800217975, 0.5889016456180636)
    # tp_total, fp_total, fn_total: (23082.0, 16113.0, 117301.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only_temp/' \
                       'gb_0_dice_0.835_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.2770393894606133, 0.17958727196312943, 0.6057424315233061)
    # tp_total, fp_total, fn_total: (25211.0, 16409.0, 115172.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only_temp/gb_0_dice_0.819_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.1665179313273552, 0.09636494447333367, 0.6121820979274143)
    # tp_total, fp_total, fn_total: (13528.0, 8570.0, 126855.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.830_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.405620191446509, 0.3306739420015244, 0.5244955144284004)
    # tp_total, fp_total, fn_total: (46421.0, 42085.0, 93962.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.836_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.2921383310063366, 0.19376277754428956, 0.5934288893252176)
    # tp_total, fp_total, fn_total: (27201.0, 18636.0, 113182.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only/gb_0.0_dice_0.804_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.11578901225522428, 0.06585555231046494, 0.47891628677994197)
    # tp_total, fp_total, fn_total: (9245.0, 10059.0, 131138.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only/gb_0.0_dice_0.779_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.10571641819608443, 0.05902424082688075, 0.5059843673668784)
    # tp_total, fp_total, fn_total: (8286.0, 8090.0, 132097.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.825_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.36273193008039606, 0.26932748267240336, 0.5553205551883675)
    # tp_total, fp_total, fn_total: (37809.0, 30276.0, 102574.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_only/gb_0.0_dice_0.766_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1200, show_instance=False)
    # overall dice, recall, precision: (0.09864368156449754, 0.05564270152505447, 0.4341814472714909)
    # tp_total, fp_total, fn_total: (7662.0, 9985.0, 130038.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3_version_2/' \
                       'low_resolution_complete_vessel_long_v1_v2/gb_0.0_dice_0.648_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1200, show_instance=False)
    # overall dice, recall, precision: (0.09864368156449754, 0.05564270152505447, 0.4341814472714909)
    # tp_total, fp_total, fn_total: (7662.0, 9985.0, 130038.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.830_recall_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.35157278202435216, 0.277464798420658, 0.4796941335965465)
    # tp_total, fp_total, fn_total: (40337.0, 43752.0, 105040.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.825_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.35800240943351436, 0.2749334488949421, 0.5130018482390389)
    # tp_total, fp_total, fn_total: (39969.0, 37943.0, 105408.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/high_resolution_stable/' \
                       'gb_0_dice_0.829_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.18395837192462489, 0.11271384056625189, 0.5)
    # tp_total, fp_total, fn_total: (16386.0, 16386.0, 128991.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_warm_up/gb_-0.00049_dice_0.803_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.38243114039108855, 0.4017004065292309, 0.3649259187512107)
    # tp_total, fp_total, fn_total: (58398.0, 101629.0, 86979.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_warm_up/gb_0.0_dice_0.832_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.3740545581030476, 0.3476684757561375, 0.4047746802597964)
    # tp_total, fp_total, fn_total: (50543.0, 74324.0, 94834.0)
    exit()
    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_warm_up/gb_-0.00049_dice_0.803_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1200, show_instance=False)
    # overall dice, recall, precision: (0.3836483496395371, 0.3712091621281018, 0.3969501142722043)
    # tp_total, fp_total, fn_total: (50369.0, 76521.0, 85320.0)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_warm_up/gb_0.0_dice_0.832_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1200, show_instance=False)
    # overall dice, recall, precision: (0.3688458078286793, 0.30819742204600226, 0.4592113498852493)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.803_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.3210004394762835, 0.25372651794988205, 0.4368205395419341)
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1500, show_instance=False)
    # overall dice, recall, precision: (0.29905148679673277, 0.22580667440190066, 0.44262583597324884)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_long/gb_0_dice_0.803_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.3185152468593337, 0.24058826361804136, 0.4711079981681528)
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1500, show_instance=False)
    # overall dice, recall, precision: (0.29783622847606767, 0.21693878114812973, 0.47494556671101973)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/original_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_warm_up/gb_0.0_dice_0.832_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.3740545581030476, 0.3476684757561375, 0.4047746802597964)
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1500, show_instance=False)
    # overall dice, recall, precision: (0.32630140896764065, 0.27630670202773633, 0.39838483215996334)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length_complete_vessel/denoise_low-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/' \
                       'low_resolution_complete_vessel_warm_up/gb_0.0_dice_0.832_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=3000, show_instance=False)
    # overall dice, recall, precision: (0.3757726264992284, 0.3299490290761262, 0.43637703441562575)
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=1500, show_instance=False)
    # overall dice, recall, precision: (0.3264111761374363, 0.2627078844135035, 0.4308985454710227)
    exit()

    top_dict_sample_sequence = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/' \
                               'sample_sequence/pe_v3_long_length/denoise_high-resolution'
    saved_model_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/pe_v3/high_resolution_stable/' \
                       'gb_0_dice_0.829_precision_phase_model_guided.pth'
    get_performance_model_and_dataset(saved_model_path, top_dict_sample_sequence, trim_length=4000, show_instance=False)
    # overall dice, recall, precision: (0.009183278461072633, 0.00466805989767675, 0.28047448103636646)
    exit()
