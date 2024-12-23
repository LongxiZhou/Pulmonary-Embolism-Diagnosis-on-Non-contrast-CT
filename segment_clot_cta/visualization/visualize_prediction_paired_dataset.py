import pe_dataset_management.basic_functions as basic_functions
import os
import numpy as np
import Tool_Functions.Functions as Functions
import segment_clot_cta.inference.inference_pe_v3 as inference
import segment_clot_cta.inference.ct_sequence_convert_pe_v3 as converter


top_dict_pair = '/data_disk/CTA-CT_paired-dataset'

sample_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_no_gt/' \
                  'pe_v3_long_length_complete_vessel/denoise_low-resolution'

save_top_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/sample_sequence/PE_no_gt/visualization_loop_3'


def load_model():
    model = inference.load_saved_model_guided(
        '/data_disk/pulmonary_embolism/segment_clot_on_CTA/check_point/loop_3/'
        'gb_0_dice_0.802_precision_phase_model_guided.pth', high_resolution=False)
    return model


def predict_sample(sample, model=None):

    if model is None:
        model = load_model()
    sample_with_clot = inference.predict_clot_for_sample_sequence(sample, model=model)

    clot_predict = converter.reconstruct_rescaled_ct_from_sample_sequence(
        sample_with_clot, absolute_cube_length=(7, 7, 10), key='clot_prob_mask')

    return clot_predict


def visualize_one_scan(scan_name='Z210', dpi=300, model=None, interval=1):
    print("visualize clot for scan:", scan_name)
    save_dict_detail = os.path.join(save_top_dict, 'detailed_view', scan_name)
    if os.path.exists(save_dict_detail):
        print("processed at:", save_dict_detail)
        return None

    save_path_brief_view = os.path.join(save_top_dict, 'brief_view', scan_name + '.png')

    cta_dataset, _ = basic_functions.find_patient_id_dataset_correspondence(
        scan_name=scan_name, top_dict=top_dict_pair, strip=True)

    rescaled_ct = np.load(os.path.join(cta_dataset, 'rescaled_ct-denoise', scan_name + '.npz'))['array']

    sample_sequence = Functions.pickle_load_object(
        os.path.join(sample_top_dict, scan_name + '.pickle'))['sample_sequence']

    rescaled_ct = np.clip(rescaled_ct + 0.5, 0.4, 1.2)  # (-760HU, 520HU)

    predicted_clot_mask = predict_sample(sample_sequence, model=model)
    predicted_clot_mask = np.array(predicted_clot_mask > 0.5, 'float32')

    z_list = list(set(np.where(predicted_clot_mask > 0.5)[2]))
    z_list.sort()
    if len(z_list) == 0:
        print("no clot predicted")
        return None

    roi_region = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence, key='depth_cube',
                                                                        absolute_cube_length=(7, 7, 10))
    roi_region = np.array(roi_region > 0.5, 'float32')
    for z in z_list[::interval]:
        up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_clot_mask[:, :, z], show=False)
        low_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], roi_region[:, :, z], show=False)
        image = np.concatenate((up_image, low_image), axis=0)
        Functions.image_save(image, os.path.join(save_dict_detail, str(z) + '.png'), dpi=dpi)

    z = int(np.median(z_list))
    up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_clot_mask[:, :, z], show=False)
    low_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], roi_region[:, :, z], show=False)
    image = np.concatenate((up_image, low_image), axis=0)
    Functions.image_save(image, save_path_brief_view, dpi=dpi)


def visualize_all_scan_no_gt(dpi=300, model=None, interval=1):
    if model is None:
        model = load_model()
    for scan_name in os.listdir(sample_top_dict):
        visualize_one_scan(scan_name[:-7], dpi=dpi, model=model, interval=interval)


if __name__ == '__main__':
    visualize_all_scan_no_gt()
