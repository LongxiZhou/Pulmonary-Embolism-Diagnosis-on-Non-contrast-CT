import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import Tool_Functions.Functions as Functions
import numpy as np
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import \
    reconstruct_semantic_from_sample_sequence, convert_ct_into_tubes
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
import pe_dataset_management.basic_functions as basic_functions
import os


def example_convert_ct_to_sample_sequence_then_predict_clot_paired_dataset(visible_device='1'):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    patient_id = 'patient-id-135'
    visualize_slice = 297
    resolution = (4, 4, 5)
    denoise_ = False
    exclude_center_out = True
    min_depth = 0.5
    dataset_dir_cta, dataset_dir_non = basic_functions.find_patient_id_dataset_correspondence(patient_id, strip=True)
    if denoise_:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct-denoise', patient_id + '.npz'))['array']
    else:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct', patient_id + '.npz'))['array']
    depth_array = np.load(
        os.path.join(dataset_dir_non, 'depth_and_center-line/depth_array', patient_id + '.npz'))['array']
    branch_array = np.load(
        os.path.join(dataset_dir_non, 'depth_and_center-line/blood_branch_map', patient_id + '.npz'))['array']

    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, depth_array, branch_array, absolute_cube_length=resolution,
        exclude_center_out=exclude_center_out, min_depth=min_depth)
    print("length sample sequence:", len(sample_sequence))

    extracted_patch_ = reconstruct_semantic_from_sample_sequence(
        sample_sequence, resolution, key='ct_data')

    Functions.image_show(np.clip(extracted_patch_[:, :, visualize_slice], -0.25, 0.7))

    sample_sequence_predicted = predict.predict_clot_for_sample_sequence(sample_sequence)

    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_predicted, resolution, key='clot_prob_mask')

    """
    Functions.merge_image_with_mask(
        np.clip(rescaled_ct[:, :, visualize_slice], -0.25, 0.7), predict_clot_mask[:, :, visualize_slice], show=True)
    """

    predict_clot_mask[0, 0, :] = 1
    predict_clot_mask[0, 1, :] = 0
    Functions.merge_image_with_mask(
        np.clip(rescaled_ct[:, :, visualize_slice], -0.25, 0.7), predict_clot_mask[:, :, visualize_slice], show=True)


def example_predict_clot_and_show_gt(patient_id='patient-id-135', visualize_slice_list=None, visible_device='1',
                                     model_path=None, high_resolution=True, show_no_fp=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    denoise = False

    top_dict_with_gt = '/data_disk/pulmonary_embolism_final/training_samples_with_annotation'
    if high_resolution:
        top_dict_with_gt = os.path.join(top_dict_with_gt, 'high_resolution')
    else:
        top_dict_with_gt = os.path.join(top_dict_with_gt, 'low_resolution')
    if denoise:
        top_dict_with_gt = os.path.join(top_dict_with_gt, 'pe_ready_denoise')
    else:
        top_dict_with_gt = os.path.join(top_dict_with_gt, 'pe_ready_not_denoise')

    sample_path = os.path.join(top_dict_with_gt, patient_id + '.pickle')
    sample = Functions.pickle_load_object(sample_path)

    print("sample with key:", list(sample.keys()))
    for key, value in sample.items():
        if key == 'sample_sequence':
            print("sample sequence cube with key", list(sample[key][0].keys()))
            continue
        if key == 'center_line_loc_array':
            continue
        print("key:", key, "  value:", value)

    sample_sequence = sample['sample_sequence']

    # the v2 is only trained with high resolution and simulated clots
    sample_sequence_predicted = predict.predict_clot_for_sample_sequence(sample_sequence, model_path=model_path,
                                                                         trim_length=5000)

    if high_resolution:
        resolution = (4, 4, 5)
    else:
        resolution = (7, 7, 10)

    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_predicted, resolution, key='clot_prob_mask')
    clot_gt_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_predicted, resolution, key='clot_gt_mask')

    if show_no_fp:
        fp_penalty_array = reconstruct_semantic_from_sample_sequence(
            sample_sequence_predicted, resolution, key='penalty_weight_fp', background=1)
        no_fp_penalty_array = 1 - fp_penalty_array
    else:
        no_fp_penalty_array = None

    if visualize_slice_list is None:
        visualize_slice_list = list(set(np.where(clot_gt_mask > 0.5)[2]))

    visualize_slice_list = list(np.arange(np.min(visualize_slice_list), np.max(visualize_slice_list) + 1))

    predict_clot_mask[0, 0, :] = 1
    predict_clot_mask[0, 1, :] = 0

    clot_gt_mask[0, 0, :] = 1
    clot_gt_mask[0, 1, :] = 0

    dataset_dir_cta, dataset_dir_non = basic_functions.find_patient_id_dataset_correspondence(patient_id, strip=True)
    if denoise:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct-denoise', patient_id + '.npz'))['array']
    else:
        rescaled_ct = np.load(os.path.join(dataset_dir_non, 'rescaled_ct', patient_id + '.npz'))['array']

    artery_mask = np.load(os.path.join(dataset_dir_non, 'semantics/artery_mask', patient_id + '.npz'))['array']
    vein_mask = np.load(os.path.join(dataset_dir_non, 'semantics/vein_mask', patient_id + '.npz'))['array']

    blood_region_strict = np.load(
        os.path.join(dataset_dir_non, 'secondary_semantics/blood_region_strict', patient_id + '.npz'))['array']

    analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, blood_region_strict)

    for z in visualize_slice_list:
        print(z)
        up = Functions.merge_image_with_mask(
            np.clip(rescaled_ct[:, :, z], -0.25, 0.7), predict_clot_mask[:, :, z], show=False)
        mid = Functions.merge_image_with_mask(
            np.clip(rescaled_ct[:, :, z], -0.25, 0.7), clot_gt_mask[:, :, z], show=False)

        if show_no_fp:
            down = Functions.merge_image_with_mask(
                np.clip(rescaled_ct[:, :, z], -0.25, 0.7), no_fp_penalty_array[:, :, z], show=False)
            image = np.concatenate((up, mid, down), axis=0)
        else:
            image = np.concatenate((up, mid), axis=0)

        Functions.image_save(image, path='/data_disk/pulmonary_embolism_final/pictures/segmentation_visualization/'
                                         + str(z) + '.png', dpi=200)


def visualize_clot_sample_sequence(sample_sequence, rescaled_ct, show_blood_vessel=True, interval=2, save_dict=None):
    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence, (4, 4, 5), key='clot_prob_mask')

    if show_blood_vessel:
        blood_mask = reconstruct_semantic_from_sample_sequence(
            sample_sequence, (4, 4, 5), key='depth_cube')
        blood_mask = np.array(blood_mask > 0.5, 'float32')
    else:
        blood_mask = None

    visualize_slice_list = list(set(np.where(predict_clot_mask > 0.5)[2]))

    print("length_z:", len(visualize_slice_list), "clot volume:", np.sum(np.array(predict_clot_mask > 0.5)))

    predict_clot_mask[0, 0, :] = 1
    predict_clot_mask[0, 1, :] = 1

    for z in visualize_slice_list[::interval]:
        print(z)
        up = Functions.merge_image_with_mask(
            np.clip(rescaled_ct[:, :, z], -0.25, 0.7), predict_clot_mask[:, :, z], show=False)
        if not show_blood_vessel:
            if save_dict is not None:
                Functions.image_save(up, os.path.join(save_dict, str(z) + '.png'), dpi=600)
            Functions.image_show(up)
        else:
            down = Functions.merge_image_with_mask(
                np.clip(rescaled_ct[:, :, z], -0.25, 0.7), blood_mask[:, :, z], show=False)
            image = np.concatenate((up, down), axis=0)
            if save_dict is not None:
                Functions.image_save(image, os.path.join(save_dict, str(z) + '.png'), dpi=600)
            Functions.image_show(image)


def show_clot_rad(scan_name, top_dict_dataset='/data_disk/RAD-ChestCT_dataset', model_path=None):
    from chest_ct_database.public_datasets.RAD_ChestCT_dataset import load_func_for_ct
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    path_rescaled_ct = os.path.join(top_dict_dataset, 'stack_ct_rad_format', scan_name)
    rescaled_ct = load_func_for_ct(path_rescaled_ct)

    depth_array = np.load(
        os.path.join(top_dict_dataset, 'depth_and_center-line/depth_array', scan_name))['array']
    branch_array = np.load(
        os.path.join(top_dict_dataset, 'depth_and_center-line/blood_branch_map', scan_name))['array']

    artery_mask = np.load(
        os.path.join(top_dict_dataset, 'semantics/artery_mask', scan_name))['array']

    vein_mask = np.load(
        os.path.join(top_dict_dataset, 'semantics/vein_mask', scan_name))['array']

    blood_region_strict = np.load(
        os.path.join(top_dict_dataset, 'secondary_semantics/blood_region_strict', scan_name))['array']

    sample_sequence = convert_ct_into_tubes(
        rescaled_ct, depth_array, branch_array, absolute_cube_length=(4, 4, 5),
        exclude_center_out=True, min_depth=0.5)

    sample_sequence_predicted = predict.predict_clot_for_sample_sequence(sample_sequence, model_path=model_path,
                                                                         min_depth=0.5, trim=True)

    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence_predicted, (4, 4, 5), key='clot_prob_mask')

    analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, blood_region_strict)

    visualize_slice_list = list(set(np.where(predict_clot_mask > 0.5)[2]))

    print("length_z:", len(visualize_slice_list), "clot volume:", np.sum(np.array(predict_clot_mask > 0.5)))

    predict_clot_mask[0, 0, :] = 1
    predict_clot_mask[0, 1, :] = 0

    for z in visualize_slice_list[::5]:
        print(z)
        Functions.merge_image_with_mask(
            np.clip(rescaled_ct[:, :, z], -0.25, 0.7), predict_clot_mask[:, :, z], show=True)


def analysis_clot_in_av(predict_clot_mask, artery_mask, vein_mask, blood_region_strict):
    artery_strict = blood_region_strict * artery_mask
    vein_strict = blood_region_strict * vein_mask
    clot_in_artery_strict = predict_clot_mask * artery_strict
    clot_in_vein_strict = predict_clot_mask * vein_strict

    clot_in_artery = predict_clot_mask * artery_mask
    clot_in_vein = predict_clot_mask * vein_mask

    total_clot_artery = np.sum(clot_in_artery) + 0.00001
    total_clot_vein = np.sum(clot_in_vein) + 0.00001

    total_clot_artery_strict = np.sum(clot_in_artery_strict) + 0.00001
    total_clot_vein_strict = np.sum(clot_in_vein_strict) + 0.00001

    print("a-v clot ratio:", total_clot_artery / total_clot_vein, "   total artery clot:", total_clot_artery)
    print("a-v clot ratio strict:", total_clot_artery_strict / total_clot_vein_strict,
          "   total artery clot strict:", total_clot_artery_strict)

    return (total_clot_artery / total_clot_vein, total_clot_artery), \
           (total_clot_artery_strict / total_clot_vein_strict, total_clot_artery_strict)


if __name__ == '__main__':
    #  patient = 'Z111'

    patient = 'patient-id-135'
    # patient = 'patient-id-23542055'
    # patient = 'Z173'
    name = patient + '.pickle'
    ord_sum = 0
    for char in name:
        ord_sum += ord(char)
    print(patient, 'in test', ord_sum % 5 == 0)

    with_gt = True

    if not with_gt:
        model = '/data_disk/pulmonary_embolism_final/check_point_dir/high_resolution/warm_up_simulation_only/' \
                'vi_0.014_dice_0.720_precision_phase_model_guided.pth'
    else:
        model = '/data_disk/pulmonary_embolism_final/check_point_dir/' \
                'high_resolution_with_augment/with_annotation_test_id_0/best_model_guided.pth'

    # show_clot_rad('val25752.npz', model_path=model)
    # exit()
    example_predict_clot_and_show_gt(
        patient_id=patient, visualize_slice_list=None, visible_device='0', high_resolution=True,
        model_path=model, show_no_fp=False)
