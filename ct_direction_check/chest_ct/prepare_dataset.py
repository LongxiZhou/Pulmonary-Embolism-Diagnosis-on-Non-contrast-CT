from chest_ct_database.public_datasets.RAD_ChestCT_dataset import load_func_for_ct
import format_convert.spatial_normalize as spatial_normalize
import Tool_Functions.Functions as Functions
import ct_direction_check.utlis as utlis
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


def form_image(array_3d, clip_min=-1000, clip_max=50, dtype='float16'):
    shape = np.shape(array_3d)
    image = np.zeros([3, shape[0], shape[1]], dtype=dtype)
    image[0] = np.average(array_3d, axis=0)
    image[1] = np.average(array_3d, axis=1)
    image[2] = np.average(array_3d, axis=2)
    image = np.clip(image, Functions.change_to_rescaled(clip_min), Functions.change_to_rescaled(clip_max))

    image = np.reshape(image, (1, 768, 256))

    return image


def process_one_array(rescaled_ct, shape=(256, 256, 256), clip_min=-1000, clip_max=50, augment_label_list=None):
    # each array will return 48 images
    if augment_label_list is None:
        augment_label_list = Functions.pickle_load_object(
           '/home/chuy/PythonProjects/Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT-main/list_label_augment.pickle')

    rescaled_ct = spatial_normalize.rescale_to_new_shape(rescaled_ct, shape)

    sample_list = []

    for class_id in range(48):
        augment_label = augment_label_list[class_id]
        rescaled_ct_new = utlis.random_flip_rotate_swap(rescaled_ct, deep_copy=True, labels=augment_label)
        image = form_image(rescaled_ct_new, clip_min=clip_min, clip_max=clip_max)
        sample_list.append((image, class_id))

    return sample_list


def pipeline_process(dataset='All', top_dict_save='/data_disk/chest_ct_direction/training_samples/',
                     fold=(0, 1), clip_max_50=True, wrong_list=None):

    if dataset == 'All':
        for dataset in ['rad', 'mudanjiang', 'yidayi', 'xwzc', 'four_center_data']:
            pipeline_process(dataset, top_dict_save, fold, clip_max_50, wrong_list)
        return None

    if clip_max_50:
        clip_min = -1000
        clip_max = 50
        save_dict_dataset = os.path.join(top_dict_save, 'clip_max_50HU')
    else:
        clip_min = -1000
        clip_max = 1000
        save_dict_dataset = os.path.join(top_dict_save, 'not_clip')

    top_dict_ct, top_dict_depth_and_branch = get_top_dicts(dataset, False)

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

        if dataset == 'rad':
            rescaled_ct = load_func_for_ct(os.path.join(top_dict_ct, file_name))
        else:
            if file_name[:-1] == 'y':
                rescaled_ct = np.load(os.path.join(top_dict_ct, file_name))
            else:
                rescaled_ct = np.load(os.path.join(top_dict_ct, file_name))['array']

        sample_list = process_one_array(rescaled_ct, shape=(256, 256, 256), clip_min=clip_min, clip_max=clip_max)

        Functions.pickle_save_object(os.path.join(save_dict_dataset, file_name[:-4] + '.pickle'), sample_list)
        processed_count += 1


if __name__ == '__main__':
    # dataset in ['rad', 'mudanjiang', 'yidayi', 'xwzc', 'four_center_data']
    current_fold = (0, 10)

    pipeline_process(fold=current_fold)
    pipeline_process(clip_max_50=False, fold=current_fold)
    exit()

    process_one_array(np.load('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/four_center_data/Scanner-A-A2.npz')['array'])

    exit()
