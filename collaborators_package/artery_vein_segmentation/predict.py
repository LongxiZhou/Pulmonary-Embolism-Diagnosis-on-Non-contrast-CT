import os
import numpy as np
import torch
import yaml
import torch.nn as nn
from collaborators_package.artery_vein_segmentation.model_for_unet import get_model
from collaborators_package.artery_vein_segmentation.utils_torch import load_checkpoint
import math
from collaborators_package.artery_vein_segmentation.combine_two_stage import combine_together


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
device = torch.device("cuda:0")
patch = [128, 128, 128]
stride = [64, 64, 64]
array_shape = [512, 512, 512]
test_dict = {
    "yaml": "/home/zhoul0a/Desktop/transfer/Artery_Vein_Segmentation/test_config.yaml",
    "check_point_1": "/home/zhoul0a/Desktop/prognosis_project/check_points/Artery_Vein_Seg/Step_1.pytorch",
    "check_point_2": "/home/zhoul0a/Desktop/prognosis_project/check_points/Artery_Vein_Seg/Step_2.pytorch"}
# print(config)


def set_filepath(yaml_name, term, content):
    with open(yaml_name) as f:
        doc = yaml.safe_load(f)

    if term == "file_path":
        doc["loaders"]["test"]["file_paths"] = content
    if term == "model_guided_path":
        doc["model_guided_path"] = content

    with open(yaml_name, 'w') as f:
        yaml.safe_dump(doc, f)


def preprocess(array):
    array = array[:, :, :, np.newaxis]
    array = torch.from_numpy(array)
    array = array.to(device)
    array = array.to(torch.float)
    return array


def array_cut(np_array):
    # Here np_array is a 3D [H, W, D] array
    # print(np_array.shape)
    assert np_array.shape == (512, 512, 512)

    n_w = math.ceil(array_shape[0] / stride[0])
    n_d = math.ceil(array_shape[1] / stride[1])
    n_h = math.ceil(array_shape[2] / stride[2])

    graph_cut = np.zeros([n_w - 1, n_d - 1, n_h - 1, patch[0], patch[1], patch[2]])

    for i in range(n_w - 1):
        for j in range(n_d - 1):
            for k in range(n_h - 1):
                loc = [i * stride[0], j * stride[1], k * stride[2]]
                graph_cut[i, j, k] = np_array[loc[0]:loc[0] + patch[0], loc[1]:loc[1] + patch[1], loc[2]:loc[2] + patch[2]]

    return graph_cut


def array_assembly(array):
    # Here prediction_array is (7, 7, 7, 2, 128, 128, 128)
    # convert it to (512, 512, 512)
    assert array.shape == (7, 7, 7, 2, 128, 128, 128)
    prediction = np.zeros([2, 512, 512, 512])

    n_w = math.ceil(array_shape[0] / stride[0])
    n_d = math.ceil(array_shape[1] / stride[1])
    n_h = math.ceil(array_shape[2] / stride[2])

    for i in range(n_w - 2):
        for j in range(n_d - 2):
            for k in range(n_h - 2):
                loc = [i * stride[0], j * stride[1], k * stride[2]]
                # print(prediction[loc[0]:loc[0] + patch[0], loc[1]:loc[1] + patch[1], loc[2]:loc[2] + patch[2]].shape)
                prediction[:, loc[0]:loc[0] + patch[0], loc[1]:loc[1] + patch[1], loc[2]:loc[2] + patch[2]] += array[i, j, k]

    return prediction / 8


def sigmoid(img, alpha, beta):

    img_max = 255
    return img_max / (1 + np.exp((beta - img) / alpha))


def predict_one_stage(graph_cut, model, prediction_result, batch=4):

    graph_cut = preprocess(graph_cut)
    graph_cut = torch.flatten(graph_cut, 0, 2)
    iter_n = graph_cut.shape[0] // batch

    for i in range(iter_n + 1):
        begin = i * batch
        end = min((i + 1) * batch, graph_cut.shape[0])
        sub_array = graph_cut[begin:end]
        sub_prediction = model(sub_array)

        for k in range(begin, end):
            x = k // (7 * 7)
            y = (k - 49 * x) // 7
            z = k - 49 * x - 7 * y
            prediction_result[x, y, z] = sub_prediction[k - begin].cpu().detach().numpy()

    return array_assembly(prediction_result)


def predict_artery_and_vein(rescaled_ct, batch_size=4, show=False):
    """
    :param show:
    :param batch_size:
    :param rescaled_ct: rescaled array with shape [512, 512, 512]
    :return: artery and vein, binary array in shape [512, 512, 512], numpy float32
    """
    rescaled_ct = (1600 * rescaled_ct + 400) / 1400
    rescaled_ct[rescaled_ct > 1] = 1
    rescaled_ct[rescaled_ct < 0] = 0

    graph_cut = array_cut(rescaled_ct)
    prediction_result = np.zeros([7, 7, 7, 2, 128, 128, 128])
    set_filepath(yaml_name=test_dict["yaml"], term="model_guided_path", content=test_dict["check_point_1"])
    config = yaml.safe_load(open(test_dict["yaml"], 'r'))
    model = get_model(config)

    if torch.cuda.device_count() > 1:
        if show:
            print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        if show:
            print("Using only single GPU")

    checkpoint = config["model_guided_path"]
    if show:
        print(checkpoint)
    model = load_checkpoint(checkpoint, model)
    model = model.to(device)
    model = model.to(torch.float)

    prediction_stage_1 = predict_one_stage(graph_cut, model, prediction_result, batch_size)

    rescaled_ct = sigmoid(rescaled_ct * 255, 20, 80) / 255
    graph_cut = array_cut(rescaled_ct)
    set_filepath(yaml_name=test_dict["yaml"], term="model_guided_path", content=test_dict["check_point_2"])
    config = yaml.safe_load(open(test_dict["yaml"], 'r'))
    checkpoint = config["model_guided_path"]
    if show:
        print(checkpoint)
    model = load_checkpoint(checkpoint, model)
    model = model.to(device)
    model = model.to(torch.float)

    prediction_stage_2 = predict_one_stage(graph_cut, model, prediction_result, batch_size)

    return combine_together(prediction_stage_1, prediction_stage_2, connect_num=3)


if __name__ == '__main__':
    exit()
    rescaled_array = np.load('/home/zhoul0a/Desktop/absolutely_normal/rescaled_ct/Scanner-A_A1.npy')
    artery, vein = predict_artery_and_vein(rescaled_array)
    import visualization.visualize_3d.visualize_stl as stl
    import Tool_Functions.Functions as Functions
    stl.visualize_numpy_as_stl(artery)
    stl.visualize_numpy_as_stl(vein)
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), artery[:, :, 256])
    Functions.merge_image_with_mask(np.clip(rescaled_array[:, :, 256] + 0.5, 0, 1), vein[:, :, 256])
    exit()
