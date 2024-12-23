import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import numpy as np
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import \
    reconstruct_semantic_from_sample_sequence
import Tool_Functions.Functions as Functions


def analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, show=True):

    artery_volume = np.sum(artery_mask) + 0.001
    vein_volume = np.sum(vein_mask) + 0.001

    clot_in_artery = predict_clot_mask * artery_mask
    clot_in_vein = predict_clot_mask * vein_mask

    ratio_clot_artery = np.sum(clot_in_artery) / artery_volume
    total_artery_clot_volume = ratio_clot_artery * artery_volume * (334 / 512 * 334 / 512)
    ratio_clot_vein = np.sum(clot_in_vein) / vein_volume
    total_vein_clot_volume = ratio_clot_vein * vein_volume * (334 / 512 * 334 / 512)

    avr = ratio_clot_artery / ratio_clot_vein
    acv = total_artery_clot_volume
    vcv = total_vein_clot_volume
    acr = ratio_clot_artery
    vcr = ratio_clot_vein

    if show:
        print("a-v clot ratio:", avr, "   artery clot ratio:", acr,
              "   artery clot volume in mm^3", acv, '\n',
              "vein clot ratio:", vcr, "   vein clot volume:", vcv)

    return {'avr': avr, 'acr': acr, 'acv': acv, 'vcr': vcr, 'vcv': vcv}


def predict_on_sample_sequence(sample_sequence, model=None, model_path=None, trim_length=4000, return_clot_array=True):
    if model is None:
        model = predict.load_saved_model_guided(model_path=model_path)

    trim = False
    if trim_length is not None:
        trim = True
    sample_sequence_with_predict = predict.predict_clot_for_sample_sequence(
        sample_sequence, model=model, min_depth=0.5, trim=trim, trim_length=trim_length)

    if return_clot_array:
        return reconstruct_semantic_from_sample_sequence(
            sample_sequence_with_predict, (4, 4, 5), key='clot_prob_mask', background=0)

    return sample_sequence_with_predict


def predict_on_evaluate_sample(sample_evaluate, model=None, model_path=None, trim_length=4000, visualize=True,
                               show_statistic=True):

    sample_sequence = sample_evaluate['sample_sequence']

    sample_sequence_with_predict = predict_on_sample_sequence(sample_sequence, model, model_path, trim_length,
                                                              return_clot_array=False)

    clot_prob_v0 = reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_predict, (4, 4, 5), key='clot_prob_mask', background=0)

    artery_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_predict, (4, 4, 5), key='artery_mask', background=0)

    vein_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_with_predict, (4, 4, 5), key='vein_mask', background=0)

    if visualize:
        ct_array = reconstruct_semantic_from_sample_sequence(
            sample_sequence_with_predict, (4, 4, 5), key='ct_data', background=0)
        ct_array = np.clip(ct_array, -0.5, 0.7)

        _, z = Functions.get_max_projection(clot_prob_v0, projection_dim=2)

        Functions.merge_image_with_mask(ct_array[:, :, z], clot_prob_v0[:, :, z])

    return analysis_clot_in_av(clot_prob_v0, artery_mask, vein_mask, show=show_statistic)


if __name__ == '__main__':
    import os
    Functions.set_visible_device('0')
    top_dict = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation_cta_confirm/' \
               'pe_vessel_high_recall/high_resolution/pe_not_trim_not_denoise/'
    fn_list = os.listdir(top_dict)
    test_set = []
    for fn in fn_list:
        if Functions.get_ord_sum(fn) % 5 == 0 and 'patient-id' in fn:
            test_set.append(fn)
    name = test_set[2]
    print(name)
    sample = Functions.pickle_load_object(top_dict + name)
    print(Functions.get_ord_sum(name) % 5)
    predict_on_evaluate_sample(sample, model_path='/home/zhoul0a/Desktop/transfer/temp/'
                                                  'vi_0.014_dice_0.635_precision_phase_model_guided.pth')
