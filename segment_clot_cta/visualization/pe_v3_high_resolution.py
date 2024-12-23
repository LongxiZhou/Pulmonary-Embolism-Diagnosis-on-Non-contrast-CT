"""
given a sequence (list of dict)
"""
import os
from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided, predict_clot_for_sample_sequence


def clip_ct_data(sample_sequence):
    import copy
    sample_sequence_new = copy.deepcopy(sample_sequence)

    def clip_signal(ct_data):
        """

        :param ct_data:
        :return: rescaled_ct_clipped (region < -200 HU set to -600 HU, region > 200 HU, set to 200 HU)
        """
        region_greater_than_minus_200 = np.array(ct_data > Functions.change_to_rescaled(-200), 'float32')
        ct_data = ct_data * region_greater_than_minus_200
        ct_data = np.clip(ct_data, 0, Functions.change_to_rescaled(200))
        return ct_data

    for item in sample_sequence_new:
        item['ct_data'] = clip_signal(item['ct_data'])

    return sample_sequence_new


def visualize_prediction_for_one_file(top_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/',
                                      file_name='patient-id-135', gt=False, loaded_model=None):
    """

    :param top_dict: dataset top dict
    :param file_name:
    :param gt: whether we have ground truth for this dataset
    :param loaded_model:
    :return: None
    """

    save_top_dict = top_dict + 'visualization/predict_clot/' + file_name + '/'
    if os.path.exists(save_top_dict):
        print("processed")
        return None

    sample_sequence_ = Functions.pickle_load_object(
        top_dict + 'sample_sequence/pe_v3_long_length/denoise_high-resolution/' +
        file_name + '.pickle')["sample_sequence"]

    print(len(sample_sequence_))

    roi_region = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence_, key='depth_cube')
    roi_region = np.array(roi_region > 0.25, 'float32')

    predict_clot_for_sample_sequence(sample_sequence_, model=loaded_model, high_resolution=True)
    probability_mask = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_sequence_, (4, 4, 5),
                                                                              key='clot_prob_mask')

    print(np.max(probability_mask))

    rescaled_ct = np.load(
        top_dict + 'rescaled_ct-denoise/' + file_name + '.npz')['array']

    rescaled_ct = np.clip(rescaled_ct + 0.5, 0, 1.2)

    if gt is not False:
        gt_clot = np.load(top_dict + 'rescaled_gt/' + file_name + '.npz')['array']
    else:
        gt_clot = None

    predicted_mask = np.array(probability_mask > 0.5, 'float32')

    loc_array = np.where(predicted_mask > 0)
    z_list = list(set(loc_array[2]))
    z_list.sort()

    for z in z_list[::2]:
        up_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], predicted_mask[:, :, z], show=False)
        low_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], roi_region[:, :, z], show=False)
        if gt_clot is not None:
            gt_image = Functions.merge_image_with_mask(rescaled_ct[:, :, z], gt_clot[:, :, z], show=False)
            image = np.concatenate((gt_image, up_image, low_image), axis=0)
        else:
            image = np.concatenate((up_image, low_image), axis=0)
        Functions.image_save(image, top_dict + 'visualization/predict_clot (high_reso)/' +
                             file_name + '/' + str(z) + '.png',
                             dpi=300)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    import Tool_Functions.Functions as Functions
    import numpy as np
    import pulmonary_embolism_v3.utlis.sequence_rescaled_ct_converter as converter

    fn_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_with_gt/rescaled_ct-denoise/')
    for fn in fn_list[::2]:
        visualize_prediction_for_one_file(file_name=fn[:-4], gt=True,
                                          loaded_model=load_saved_model_guided(high_resolution=True))
