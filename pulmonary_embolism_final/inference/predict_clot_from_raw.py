import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import Tool_Functions.Functions as Functions
import format_convert.dcm_np_converter_new as dcm_to_np
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
import pulmonary_embolism.prepare_dataset.get_branch_mask as get_branch_mask
import basic_tissue_prediction.predict_rescaled as predict_semantics
import collaborators_package.artery_vein_segmentation_v2.longxi_adaptation as seg_artery_vein_airway
import analysis.center_line_and_depth_3D as depth_and_center_line
import analysis.other_functions as other_functions
from pulmonary_embolism_final.utlis.other_funcs import get_rank_count
from pulmonary_embolism_final.performance.get_av_classification_metrics import process_to_get_metrics
import ct_direction_check.chest_ct.inference as direction_inference
import os


if __name__ == '__main__':

    # please modify
    top_dir_models = '/home/zhoul0a/Desktop/Out_Share_PE/Data_and_Models'

    # please modify
    Functions.set_visible_device('1')

    # dcm files dir, please modify
    rescaled_ct = dcm_to_np.establish_rescale_chest_ct(os.path.join(top_dir_models, 'example_data/cjw'))

    # Make sure the direction is current, otherwise the model will fail
    rescaled_ct = direction_inference.cast_to_standard_direction(
        rescaled_ct, model_path=os.path.join(top_dir_models, 'direction_normalize.pth'))

    lung_mask = predict_semantics.predict_lung_masks_rescaled_array(
        rescaled_ct, check_point_top_dict=os.path.join(top_dir_models, 'semantic_models'))

    blood_vessel_mask = predict_semantics.get_prediction_blood_vessel(
        rescaled_ct, lung_mask=lung_mask, check_point_top_dict=os.path.join(top_dir_models, 'semantic_models'))

    artery, vein = seg_artery_vein_airway.predict_av_rescaled(
        rescaled_ct, lung_mask=lung_mask, model_path=os.path.join(
            top_dir_models, 'semantic_models', 'chest_segmentation', 'predict_av_main_3_unzip.pth'))

    blood_vessel_mask = other_functions.smooth_mask(blood_vessel_mask)
    depth_array = depth_and_center_line.get_surface_distance(blood_vessel_mask, strict=True)
    blood_center_line = depth_and_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)

    branch_map = get_branch_mask.get_branching_cloud(blood_center_line, depth_array, search_radius=5,
                                                     smooth_radius=1,
                                                     step=1, weight_half_decay=20, refine_radius=4)

    bgvt_model = predict.load_saved_model_guided(model_path=os.path.join(top_dir_models, 'BGVT_model.pth'))

    metrics, sample_sequence = process_to_get_metrics(
        rescaled_ct, depth_array, artery, vein, branch_map, blood_region_strict=None,
        model=bgvt_model, return_sequence=True)

    metric_value = metrics['v0']

    from pulmonary_embolism_final.inference.posterior_pe_prob import posterior_prob

    positive_rer_dis = Functions.pickle_load_object(os.path.join(top_dir_models, 'rer_for_PE_positives.pickle'))
    negative_rer_dis = Functions.pickle_load_object(os.path.join(top_dir_models, 'rer_for_PE_negatives.pickle'))

    print("\n\nIf this patient already achieved criteria for CTPA, it has:",
          posterior_prob(metric_value, positive_rer_dis, negative_rer_dis),
          "probability to be PE positive")

    distribution_list = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/Out_Share_PE/Data_and_Models/distribution_non_PE.pickle')

    rank = get_rank_count(metric_value, distribution_list)

    print("\n\nhigher a-v clot ratio means more possibility of pulmonary embolism")
    print("in our data set of 4737 non-PE CT, this scan is higher than:", rank / 4737 * 100, '% of these scans')
