import numpy as np
import format_convert.basic_transformations as basic_transform
from collaborators_package.chest_register.inference.inference import normalization, register_with_given_flow, register
import Tool_Functions.Functions as Functions
import os


def compute_dice(pre, mask):
    return 2 * np.sum(pre * mask) / (np.sum(pre * pre) + np.sum(mask * mask))


def get_comparisons(patient_id='Z103.npz', simulated_non_contrast=False, normalize=False, depth_guide=False,
                    show_stl=False, show_detail=False):

    if simulated_non_contrast is True:
        rescaled_fix = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                               'simulated_non_contrast/rescaled_ct-denoise/' + patient_id)['array']

        vessel_fixed = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                               'simulated_non_contrast/semantics/blood_mask_high_recall/' + patient_id)['array']
    else:
        rescaled_fix = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                               'rescaled_ct-denoise/' + patient_id)['array']
        vessel_fixed = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                               'semantics/blood_mask_high_recall/' + patient_id)['array']

    rescaled_moving = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/'
                              'PE_High_Quality/rescaled_ct-denoise/' + patient_id)['array']
    vessel_moving = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/'
                            'semantics/blood_mask_high_recall/' + patient_id)['array']

    if show_detail:
        rescaled_fix_cta = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                                   'rescaled_ct-denoise/' + patient_id)['array']
        rescaled_moving_non = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/'
                                      'PE_High_Quality/rescaled_ct-denoise/' + patient_id)['array']

    if depth_guide:
        from analysis.center_line_and_depth_3D import get_surface_distance
        depth_fix, max_depth_fix = get_surface_distance(vessel_fixed, strict=True, return_max_distance=True)
        depth_moving, max_depth_moving = get_surface_distance(vessel_moving, strict=True, return_max_distance=True)
        rescaled_fix = rescaled_fix + depth_fix / max_depth_fix
        rescaled_moving = rescaled_moving + depth_moving / max_depth_moving

    if normalize:
        vessel_fixed, flow_fix = normalization(vessel_fixed)
        vessel_moving, flow_moving = normalization(vessel_moving)
        rescaled_fix = basic_transform.transformation_on_array(rescaled_fix, flow_fix)
        rescaled_moving = basic_transform.transformation_on_array(rescaled_moving, flow_moving)

        if show_detail:
            rescaled_fix_cta = basic_transform.transformation_on_array(rescaled_fix_cta, flow_fix)
            rescaled_moving_non = basic_transform.transformation_on_array(rescaled_moving_non, flow_moving)

    if show_stl:
        import visualization.visualize_3d.visualize_stl as stl
        stl.visualize_numpy_as_stl(vessel_fixed)
        stl.visualize_numpy_as_stl(vessel_moving)

    original_dice = compute_dice(vessel_moving, vessel_fixed)

    if not normalize:
        registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow = \
            register(rescaled_moving, rescaled_fix, vessel_moving, vessel_fixed, two_stage=False, return_flow=True)
    else:
        registered_ct, registered_seg, fixed_ct, fixed_seg, register_flow = \
            register(rescaled_moving, rescaled_fix, vessel_moving, vessel_fixed, two_stage=False, down_sample=False,
                     return_flow=True)

    print(np.shape(fixed_seg), 'fix_seg')
    print(np.shape(register_flow), 'register_flow')

    registered_dice = compute_dice(fixed_seg, registered_seg)

    if show_detail:
        print(original_dice, registered_dice)

    if show_detail:
        registered_non = register_with_given_flow(rescaled_moving_non, register_flow)
        for i in range(50, 250, 1):
            image_left = registered_non[:, :, i]
            image_right = rescaled_fix_cta[:, :, i]
            image = np.concatenate([image_left, image_right], axis=1)
            image = np.clip(image + 0.5, -0.1, 1.2)
            Functions.image_show(image, gray=True)
            # Functions.image_save(image, '/data_disk/temp/transfer/' + 'Z138_' + str(i) + '.png', gray=True, dpi=300)

    return original_dice, registered_dice


def ablation_study_for_conditions():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    report_save_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/reports/' \
                       'registration_dice_paired_CTA_non-contrast.pickle'

    fn_set_1 = set(os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                              'simulated_non_contrast/rescaled_ct-denoise/'))
    fn_set_2 = set(os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                              'rescaled_ct-denoise/'))

    fn_list = list(fn_set_1 & fn_set_2)

    if os.path.exists(report_save_path):
        report_dict = Functions.pickle_load_object(report_save_path)
        processed_list = list(report_dict.keys())
    else:
        processed_list = []
        report_dict = {}

    print(processed_list)

    for fn in Functions.iteration_with_time_bar(fn_list):
        try:
            print("processing:", fn)
            if fn in processed_list:
                print("processed")
                continue

            condition_1 = get_comparisons(fn, simulated_non_contrast=False, normalize=False, depth_guide=False)
            condition_2 = get_comparisons(fn, simulated_non_contrast=True, normalize=False, depth_guide=False)
            condition_3 = get_comparisons(fn, simulated_non_contrast=True, normalize=True, depth_guide=False)
            condition_4 = get_comparisons(fn, simulated_non_contrast=True, normalize=True, depth_guide=True)

            case_dict = {"1": condition_1, "2": condition_2, "3": condition_3, "4": condition_4}
            print(case_dict)
            report_dict[fn] = case_dict

            Functions.pickle_save_object(report_save_path, report_dict)
        except:
            continue


def show_ablation_results():
    report_save_path = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/reports/' \
                       'registration_dice_paired_CTA_non-contrast.pickle'

    report_dict = Functions.pickle_load_object(report_save_path)

    condition_1 = []
    condition_2 = []
    condition_3 = []
    condition_4 = []

    for key in list(report_dict.keys()):
        print(key, report_dict[key])
        condition_1.append(report_dict[key]['1'])
        condition_2.append(report_dict[key]['2'])
        condition_3.append(report_dict[key]['3'])
        condition_4.append(report_dict[key]['4'])

    print(np.average(condition_1), np.std(condition_1))
    print(np.average(condition_2), np.std(condition_2))
    print(np.average(condition_3), np.std(condition_3))
    print(np.average(condition_4), np.std(condition_4))

    return condition_1, condition_2, condition_3, condition_4


if __name__ == '__main__':
    show_ablation_results()
    exit()
    get_comparisons('Z138.npz', simulated_non_contrast=True, normalize=True, depth_guide=True,
                    show_stl=False, show_detail=False)
    exit()
    show_ablation_results()
