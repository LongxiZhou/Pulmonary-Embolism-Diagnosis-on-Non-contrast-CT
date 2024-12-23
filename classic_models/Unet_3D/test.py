import torch
import classic_models.Unet_3D.U_net_Model_3D as unm


def load_model(check_point_path, array_info, gpu=True):
    if array_info["encoders"] == 4:
        model = unm.UNet3D(in_channels=array_info["channels_data"], out_channels=array_info["channels_out"],
                           init_features=array_info["init_features"])
    elif array_info["encoders"] == 3:
        model = unm.UNet3DSimple(in_channels=array_info["channels_data"], out_channels=array_info["channels_out"],
                                 init_features=array_info["init_features"])
    elif array_info["encoders"] == 2:
        model = unm.UNet3DSimplest(in_channels=array_info["channels_data"], out_channels=array_info["channels_out"],
                                   init_features=array_info["init_features"])

    model_dict = torch.load(check_point_path)["state_dict"]
    if not array_info["mute_output"]:
        print("loading checkpoint")
    model.load_state_dict(model_dict)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if gpu is False:
        device = "cpu"
    if torch.cuda.device_count() > 1 and gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if not array_info["mute_output"]:
        print("checkpoint loaded with", torch.cuda.device_count(), 'GPU')
    return model


def predict_one_sample(sample, model):
    """
    :param:
    :return: the predicted array, float32 in shape [batch_size, x, y, semantic_channel]
    """

    model.eval()
    with torch.no_grad():
        sample = torch.tensor(sample, requires_grad=False).float().cuda()
        predict = model(sample)
        positives = (predict[:, 1, :, :, :] > predict[:, 0, :, :, :]).float().unsqueeze(1)
        num_positives = positives.sum().float().item()
        positive_to_roi_ratio = num_positives / sample[2, :, :, :].sum().float().item()

    if num_positives > 27:
        contain_nodule = True
    else:
        contain_nodule = False

    return contain_nodule, positive_to_roi_ratio
