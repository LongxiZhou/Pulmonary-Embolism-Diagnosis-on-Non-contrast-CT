from registration_pulmonary.models.model_no_landmark import RegisterWithGivenFlow
import torch
import numpy as np


def register_with_given_flow(numpy_array, flow_on_cpu, device=torch.device("cuda")):

    tensor_moving = torch.FloatTensor(numpy_array).to(device)
    flow_on_gpu = torch.FloatTensor(flow_on_cpu).to(device)
    transform_with_flow = RegisterWithGivenFlow(np.shape(numpy_array)[2:])
    transform_with_flow = transform_with_flow.to(device)

    registered_tensor = transform_with_flow(tensor_moving, flow_on_gpu)

    return registered_tensor.cpu().detach().numpy()[b, 0]


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    b = 0

    array = np.zeros([2, 1, 100, 100, 100], 'float32')

    array[0, 0, 25: 75, 25: 75, 25: 75] = \
        np.load('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/xwzc/xwzc000024.npz')['array'][
        200: 250,
        200: 250, 200: 250]

    array[1, 0, 25: 75, 25: 75, 25: 75] = \
        np.load('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/xwzc/xwzc000024.npz')['array'][
        200: 250,
        200: 250, 200: 250]

    flow = np.zeros([2, 3, 100, 100, 100], 'float32')

    flow[0, 0] = np.arange(0, 100) / 3
    flow[0, 1] = np.arange(0, 100) / 5
    flow[0, 2] = np.arange(0, 100) / 10

    new_array = register_with_given_flow(array, flow)

    Functions.image_show(array[b, 0, 50, :, :])
    Functions.image_show(new_array[50, :, :])
    Functions.image_show(flow[b, 1, 50, :, :])
    Functions.image_show(flow[b, 2, 50, :, :])

    Functions.image_show(array[b, 0, :, 50, :])
    Functions.image_show(new_array[:, 50, :])
    Functions.image_show(flow[b, 1, :, 50, :])
    Functions.image_show(flow[b, 2, :, 50, :])

    Functions.image_show(array[b, 0, :, :, 50])
    Functions.image_show(new_array[:, :, 50])
    Functions.image_show(flow[b, 0, :, :, 50])
    Functions.image_show(flow[b, 1, :, :, 50])
    exit()
