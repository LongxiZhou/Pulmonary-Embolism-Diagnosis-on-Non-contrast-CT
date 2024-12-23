from chest_ct_database.public_datasets.RAD_ChestCT_dataset import load_func_for_ct
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import convert_ct_into_tubes
import Tool_Functions.Functions as Functions
import numpy as np
import os


def get_top_dicts(dataset='rad', denoise=True):

    if dataset == 'rad':
        if denoise:
            top_dict_ct = '/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise/'
        else:
            top_dict_ct = '/data_disk/RAD-ChestCT_dataset/stack_ct_rad_format/'
        top_dict_depth_and_branch = \
            '/data_disk/RAD-ChestCT_dataset/depth_and_center-line/'

    elif dataset == 'mudanjiang':
        if denoise:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/COVID-19/mudanjiang/'
        else:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct/COVID-19/mudanjiang/'
        top_dict_depth_and_branch = \
            '/data_disk/rescaled_ct_and_semantics/depth_and_center-line/COVID-19/mudanjiang/'

    elif dataset == 'yidayi':
        if denoise:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/COVID-19/yidayi/'
        else:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct/COVID-19/yidayi/'
        top_dict_depth_and_branch = \
            '/data_disk/rescaled_ct_and_semantics/depth_and_center-line/COVID-19/yidayi/'

    elif dataset == 'four_center_data':
        if denoise:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/healthy_people/four_center_data/'
        else:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/four_center_data/'
        top_dict_depth_and_branch = \
            '/data_disk/rescaled_ct_and_semantics/depth_and_center-line/healthy_people/four_center_data/'

    else:
        assert dataset == 'xwzc'
        if denoise:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct-denoise/healthy_people/xwzc/'
        else:
            top_dict_ct = '/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/xwzc/'
        top_dict_depth_and_branch = \
            '/data_disk/rescaled_ct_and_semantics/depth_and_center-line/healthy_people/xwzc/'

    return top_dict_ct, top_dict_depth_and_branch


def pipeline_process_not_pe(high_resolution=False, fold=(0, 1), wrong_list=None, dataset='All', denoise=True,
                            for_evaluation=False):

    if dataset == 'All':
        for dataset in ['rad', 'mudanjiang', 'yidayi', 'xwzc', 'four_center_data']:
            pipeline_process_not_pe(high_resolution, fold, wrong_list, dataset, denoise, for_evaluation)
        return None

    if not high_resolution:
        absolute_cube_length = (7, 7, 10)
        min_depth = 2.5
        exclude_center_out = False
    else:
        absolute_cube_length = (4, 4, 5)
        min_depth = 0.5
        exclude_center_out = True

    top_dict_save = '/data_disk/pulmonary_embolism_final/training_samples_simulate_clot/'
    if for_evaluation:
        top_dict_save = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation/non_pe'

    if high_resolution:
        top_dict_save = os.path.join(top_dict_save, 'high_resolution')
    else:
        top_dict_save = os.path.join(top_dict_save, 'low_resolution')

    if denoise:
        save_dict_dataset = os.path.join(top_dict_save, 'not_pe_not_trim_denoise')
    else:
        save_dict_dataset = os.path.join(top_dict_save, 'not_pe_not_trim_not_denoise')

    top_dict_ct, top_dict_depth_and_branch = get_top_dicts(dataset, denoise)

    list_file_name = os.listdir(top_dict_ct)[fold[0]::fold[1]]

    if wrong_list is None:
        wrong_list = []

    processed_count = 0
    for file_name in list_file_name:
        if file_name in wrong_list:
            print("wrong scan")
            processed_count += 1
            continue
        print("\nprocessing:", file_name, len(list_file_name) - processed_count, 'left')
        if os.path.exists(os.path.join(save_dict_dataset, file_name[:-4] + '.pickle')):
            print('processed')
            processed_count += 1
            continue

        if dataset == 'rad' and denoise is False:
            rescaled_ct = load_func_for_ct(os.path.join(top_dict_ct, file_name))
        else:
            if file_name[:-1] == 'y':
                rescaled_ct = np.load(os.path.join(top_dict_ct, file_name))
            else:
                rescaled_ct = np.load(os.path.join(top_dict_ct, file_name))['array']

        path_depth_array = os.path.join(top_dict_depth_and_branch, 'depth_array', file_name[:-4] + '.npz')
        depth_array = np.load(path_depth_array)['array']

        path_branch_array = os.path.join(top_dict_depth_and_branch, 'blood_branch_map', file_name[:-4] + '.npz')
        branch_array = np.load(path_branch_array)['array']

        blood_center_line_path = os.path.join(top_dict_depth_and_branch, 'blood_center_line', file_name[:-4] + '.npz')
        center_line_mask = np.load(blood_center_line_path)['array']

        if for_evaluation:
            top_dict_semantic = top_dict_depth_and_branch.replace('depth_and_center-line', 'semantics')
            artery_path = os.path.join(top_dict_semantic, 'artery_mask', file_name[:-4] + '.npz')
            artery_mask = np.load(artery_path)['array']
            vein_path = os.path.join(top_dict_semantic, 'vein_mask', file_name[:-4] + '.npz')
            vein_mask = np.load(vein_path)['array']

            semantic_dict = {"artery_mask": artery_mask, "vein_mask": vein_mask}
        else:
            semantic_dict = {}

        if np.sum(center_line_mask) < 1000 or np.max(depth_array) < 15:
            print("wrong seg")
            processed_count += 1
            continue

        sample_list = convert_ct_into_tubes(
            rescaled_ct, depth_array, branch_array, absolute_cube_length=absolute_cube_length, min_depth=min_depth,
            exclude_center_out=exclude_center_out, **semantic_dict)

        print("sample list has:", len(sample_list), "elements")

        center_line_loc_array = np.where(center_line_mask > 0.5)

        sample_final = {"sample_sequence": sample_list, "center_line_loc_array": center_line_loc_array,
                        "is_PE": False, "has_clot_gt": None, "clot_gt_volume_sum": None}

        Functions.pickle_save_object(os.path.join(save_dict_dataset, file_name[:-4] + '.pickle'), sample_final)
        processed_count += 1


if __name__ == '__main__':
    # dataset in ['rad', 'mudanjiang', 'yidayi', 'xwzc', 'four_center_data']
    current_fold = (0, 4)
    Functions.set_visible_device('1')

    pipeline_process_not_pe(high_resolution=True, fold=current_fold, denoise=False, for_evaluation=True)
    exit()

    pipeline_process_not_pe(high_resolution=False, fold=current_fold, denoise=False)
    pipeline_process_not_pe(high_resolution=True, fold=current_fold, denoise=False)
    pipeline_process_not_pe(high_resolution=False, fold=current_fold, denoise=True)
    pipeline_process_not_pe(high_resolution=True, fold=current_fold, denoise=True)

    exit()
