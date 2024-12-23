import numpy as np
import Tool_Functions.Functions as Functions
from analysis.connectivity_refine_fast import select_region
from analysis.get_surface_rim_adjacent_mean import get_surface


def show_case_1():
    patient_id = '11.17p05.npz'  # 'Z138.npz' #

    rescaled_cta = np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/rescaled_ct-denoise/' +
                           patient_id)['array']

    rescaled_non_contrast = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/'
                                    'rescaled_ct-denoise/' + patient_id)['array']
    difference = rescaled_cta - rescaled_non_contrast

    difference = np.clip(difference, 0, np.inf)
    potential_blood_region = np.array(difference > 50 / 1600, 'float32')

    rescaled_cta = np.clip(rescaled_cta, Functions.change_to_rescaled(-300), Functions.change_to_rescaled(300))
    rescaled_non_contrast = np.clip(
        rescaled_non_contrast, Functions.change_to_rescaled(-300), Functions.change_to_rescaled(300))

    blood_mask_non = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/'
                             'semantics/blood_mask/' + patient_id)['array']
    blood_mask_non = blood_mask_non + get_surface(blood_mask_non, outer=True, strict=False)
    blood_mask_non = blood_mask_non * np.array(rescaled_non_contrast > Functions.change_to_rescaled(-200))

    blood_mask_high_recall_cta = np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/'
                                         'semantics/blood_mask_high_recall/' + patient_id)['array']

    blood_mask_cta = np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/'
                             'simulated_non_contrast/semantics/blood_mask/' + patient_id)['array']

    blood_mask_cta = blood_mask_cta + get_surface(blood_mask_cta, outer=True, strict=False)
    blood_mask_cta = blood_mask_cta * np.array(rescaled_cta > Functions.change_to_rescaled(-200))

    blood_mask_high_recall = np.clip(blood_mask_high_recall_cta + blood_mask_non, 0, 1)

    z_cta = 257
    z_non_contrast = 257

    blood_region = blood_mask_high_recall * potential_blood_region
    blood_region = blood_region * np.array(rescaled_cta > Functions.change_to_rescaled(50))

    blood_region = select_region(blood_region, leave_count=1)

    branch_map_non = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/'
                             'depth_and_center-line/blood_branch_map/' + patient_id)['array']
    blood_region_non = np.clip(blood_region + np.array(branch_map_non > 4), 0, 1)
    blood_region_non = blood_region_non + get_surface(blood_region, outer=True, strict=False)

    blood_region_non = blood_region_non * np.array(rescaled_non_contrast > Functions.change_to_rescaled(-200))
    blood_region_non = blood_region_non - get_surface(blood_region_non, outer=False, strict=False)
    blood_region_non = blood_region_non * blood_mask_non

    blood_mask_non = blood_mask_non + get_surface(blood_mask_non, outer=True, strict=True)
    blood_mask_non = blood_mask_non * np.array(rescaled_non_contrast > Functions.change_to_rescaled(-400))
    original_mask_non = Functions.merge_image_with_mask(
        rescaled_non_contrast[:, :, z_non_contrast], blood_mask_non[:, :, z_non_contrast], show=False)

    original_mask_cta = Functions.merge_image_with_mask(
        rescaled_cta[:, :, z_cta], blood_mask_cta[:, :, z_cta], show=False)

    blood_region_from_non = Functions.merge_image_with_mask(
        rescaled_non_contrast[:, :, z_non_contrast], blood_region_non[:, :, z_non_contrast], show=False)

    blood_region_cta = blood_region * blood_mask_cta

    blood_region_gt = Functions.merge_image_with_mask(
        rescaled_cta[:, :, z_cta], blood_region_cta[:, :, z_cta], show=False)

    inference_blood_from_non = np.concatenate((original_mask_non, blood_region_from_non), axis=0)
    input_and_gt = np.concatenate((original_mask_cta, blood_region_gt), axis=0)

    Functions.image_show(inference_blood_from_non)
    Functions.image_save(inference_blood_from_non, '/data_disk/pulmonary_embolism/segment_clot_on_CTA/'
                                                   'PE_CTA_with_gt_82/visualization/inference_blood_from_non.png',
                         dpi=300)
    Functions.image_show(input_and_gt)
    Functions.image_save(input_and_gt, '/data_disk/pulmonary_embolism/segment_clot_on_CTA/'
                                       'PE_CTA_with_gt_82/visualization/input_and_gt.png',
                         dpi=300)


if __name__ == '__main__':
    show_case_1()
