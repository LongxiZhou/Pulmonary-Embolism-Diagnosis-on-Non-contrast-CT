import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.spatial_normalize as spatial_normalize
import os
from ct_direction_check.model_cnn import AlexNet
import torch
import torch.nn as nn
from ct_direction_check.chest_ct.prepare_dataset import form_image
import ct_direction_check.utlis as utlis


augment_label_list = Functions.pickle_load_object(
    '/home/zhoul0a/Desktop/Longxi_Platform/ct_direction_check/list_label_augment.pickle')

softmax_layer = torch.nn.Softmax(dim=1)


def cast_to_standard_direction(rescaled_ct, model=None, model_path=None, show_prob=True,
                               show_image=False, deep_copy=True, return_original_direction_class=False):
    direction_class = determine_direction(rescaled_ct, model=model, model_path=model_path, show_prob=show_prob,
                                          show_image=show_image, return_probability=False)

    augment_label = augment_label_list[direction_class]

    if direction_class == 0:
        print("already in standard direction")
        if deep_copy:
            rescaled_ct_new = np.array(rescaled_ct)
        else:
            rescaled_ct_new = rescaled_ct
    else:
        print("casting direction class", direction_class, "to standard direction")
        rescaled_ct_new = utlis.random_flip_rotate_swap(
            rescaled_ct, deep_copy=deep_copy, labels=augment_label, reverse=True)

    if return_original_direction_class:
        return rescaled_ct_new, augment_label
    else:
        return rescaled_ct_new


def load_model(model_path=None):
    if model_path is None:
        model_path = '/data_disk/chest_ct_direction/check_point/all/best_model_cnn.pth'

    model = AlexNet()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")

    model = model.to("cuda:0")

    data_dict = torch.load(model_path)
    if type(model) == nn.DataParallel:
        model.module.load_state_dict(data_dict["state_dict"])
    else:
        model.load_state_dict(data_dict["state_dict"])

    return model


def sort_with_index(prediction_probability):

    def compare_func(a, b):
        if a["predicted_probability"] > b["predicted_probability"]:
            return 1
        return -1

    probability_list = []
    for i in range(len(prediction_probability)):
        probability_list.append({"predicted_probability": prediction_probability[i], "class_id": i})

    return Functions.customized_sort(probability_list, compare_func, reverse=True)


def determine_direction(rescaled_ct, model=None, model_path=None, return_probability=False, show_prob=True,
                        show_image=False, return_sorted_class_prob=False):
    rescaled_ct_reduce = spatial_normalize.rescale_to_new_shape(rescaled_ct, (256, 256, 256))
    sample_image = form_image(rescaled_ct_reduce, clip_min=-1000, clip_max=1000, dtype='float32')
    if model is None:
        model = load_model(model_path)

    sample_image = np.reshape(sample_image, (1, 1, 768, 256))
    sample_tensor = torch.FloatTensor(sample_image).cuda()

    model.eval()
    with torch.no_grad():
        prediction_probability = softmax_layer(model(sample_tensor).cpu()).numpy()[0]  # [48, ]

    sorted_class_id_probability = sort_with_index(prediction_probability)
    if show_prob:
        print("this case is predicted as class:", np.argmax(prediction_probability, axis=0))
        print("the first three predicted class are:")
        print(sorted_class_id_probability[0: 3])

    if show_image:
        Functions.image_show(sample_image[0, 0], gray=True)

    if return_probability:
        if not return_sorted_class_prob:
            return prediction_probability
        else:
            return prediction_probability, sorted_class_id_probability
    else:
        if not return_sorted_class_prob:
            return int(np.argmax(prediction_probability, axis=0))
        return int(np.argmax(prediction_probability, axis=0)), sorted_class_id_probability


def normalize_direction_dataset(top_dict='/data_disk/RSNA-PE_dataset/rescaled_ct', fold=(0, 3),
                                visible_device=None, replace_original=True):
    # modify to fit your rescaled ct dataset
    Functions.set_visible_device(visible_device)

    fn_list = Functions.split_list_by_ord_sum(os.listdir(top_dict), fold=fold)

    model_ = load_model()
    processed = 0
    for fn in fn_list:
        print("processing", fn, processed, '/', len(fn_list))
        array = np.load(os.path.join(top_dict, fn))['array']
        new_array, class_id = cast_to_standard_direction(
            array, model=model_, show_image=False, show_prob=True,
            deep_copy=False, return_original_direction_class=True)
        if not class_id == 0 and replace_original:
            print("replacing...")
            Functions.save_np_array(top_dict, fn, new_array, dtype='float16', compress=True)
        processed += 1


def form_direction_probability_dataset(fold=(0, 3)):
    # modify to fit your rescaled ct dataset
    Functions.set_visible_device('1')
    top_dict = '/data_disk/RSNA-PE_dataset/rescaled_ct'
    top_dict_save_class_id_probability = '/data_disk/RSNA-PE_dataset/pickle_objects/direction_probability'

    fn_list = os.listdir(top_dict)[fold[0]:: fold[1]]
    model_ = load_model()
    processed = 0
    for fn in fn_list:
        print("processing", fn, processed, '/', len(fn_list))
        save_path = os.path.join(top_dict_save_class_id_probability, fn[:-4] + '.pickle')
        if os.path.exists(save_path):
            print(fn, "processed")
            processed += 1
            continue

        array = np.load(os.path.join(top_dict, fn))['array']

        class_id, sorted_class_id_probability = determine_direction(
            array, model=model_, show_image=False, show_prob=True, return_probability=False,
            return_sorted_class_prob=True)

        Functions.pickle_save_object(save_path, sorted_class_id_probability)

        processed += 1


if __name__ == '__main__':
    Functions.set_visible_device('1')

    temp_array = np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/Normal_High_Quality/rescaled_ct/patient-id-20567360.npz')['array']

    cast_to_standard_direction(temp_array)
    exit()

    normalize_direction_dataset('/data_disk/CTA-CT_paired-dataset/dataset_CTA/Temp_High_Quality/rescaled_ct',
                                replace_original=False, visible_device='1')
    exit()

    form_direction_probability_dataset(fold=(0, 6))
    exit()
    normalize_direction_dataset()
