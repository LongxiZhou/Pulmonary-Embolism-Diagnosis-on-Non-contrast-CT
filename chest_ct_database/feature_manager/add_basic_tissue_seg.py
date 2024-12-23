
from pulmonary_nodules.predict_pipeline import rescaled_ct_to_semantic_seg
import numpy as np
import os


def default_load_func(path_file):
    if path_file[-1] == 'z':
        file_loaded = np.load(path_file)
        key_list = list(file_loaded.keys())
        if len(key_list) == 1:
            print("loading .npz with key:", key_list[0])
            rescaled_ct = file_loaded[key_list[0]]
            return rescaled_ct
        else:
            assert 'array' in key_list
            print("loading .npz with key:", 'array')
            rescaled_ct = file_loaded['array']
            return rescaled_ct
    print("loading npy...")
    return np.load(path_file)


def segment_varies_tissue_single_dataset(dict_rescaled_ct, top_dict_semantic, artery_vein=False, batch_size=None,
                                         fold=(0, 1),
                                         load_func=default_load_func):
    if batch_size is None:
        import torch
        batch_size = torch.cuda.device_count()
    rescaled_ct_to_semantic_seg(dict_rescaled_ct, top_dict_semantic, artery_vein, batch_size, fold,
                                load_func)


def segment_varies_tissue_database(top_dict_rescaled_ct, top_dict_semantics, artery_vein=False, batch_size=None,
                                   fold=(0, 1), load_func=default_load_func):
    if batch_size is None:
        import torch
        batch_size = torch.cuda.device_count()
    from chest_ct_database.basic_functions import extract_absolute_dirs_sub_dataset
    if not top_dict_rescaled_ct[-1] == '/':
        top_dict_rescaled_ct = top_dict_rescaled_ct + '/'

    list_dataset_dict = extract_absolute_dirs_sub_dataset(top_dict_rescaled_ct)
    print("there are", len(list_dataset_dict), 'dataset')
    print("fold:", fold)

    list_sub_dirs = []
    for dataset_dict in list_dataset_dict:
        list_sub_dirs.append(dataset_dict[len(top_dict_rescaled_ct)::])

    for dataset_sub_dir in list_sub_dirs:  # dataset_sub_dir like 'COVID-19/healthy/four_center'
        print("##########################")
        print("processing dataset:", dataset_sub_dir)
        save_dict = os.path.join(top_dict_semantics, dataset_sub_dir)
        print("saving new feature to:", save_dict)
        print("##########################")
        dict_rescaled_ct = os.path.join(top_dict_rescaled_ct, dataset_sub_dir)
        segment_varies_tissue_single_dataset(dict_rescaled_ct, save_dict, artery_vein, batch_size, fold, load_func)

    return None


if __name__ == '__main__':
    current_fold = (0, 8)
    visible_devices = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    top_dict_rescaled_ct_denoise = '/data_disk/RSNA-PE_dataset/simulated_non_contrast/rescaled_ct-denoise/'
    top_dict_save_semantics = '/data_disk/RSNA-PE_dataset/simulated_non_contrast/semantics'
    segment_varies_tissue_database(top_dict_rescaled_ct_denoise, top_dict_save_semantics,
                                   artery_vein=True, batch_size=1, fold=current_fold)
    exit()

    dict_rescaled_ct_denoise = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/denoise-rescaled_ct/'
    dict_semantic_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/dataset_embolism/semantics/'

    segment_varies_tissue_single_dataset(dict_rescaled_ct_denoise,
                                         dict_semantic_top_dict, artery_vein=True, batch_size=2, fold=(0, 3),
                                         load_func=default_load_func)
