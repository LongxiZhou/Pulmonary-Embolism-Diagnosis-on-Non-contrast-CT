import torch
import torch.nn as nn
import numpy as np
from med_transformer.position_embeding import get_3d_sincos_pos_embed_loc_list, get_4d_sincos_pos_embed_loc_list


def init_weights_vit(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def post_process_to_list(prediction_vectors, list_query_sequence, cube_shape):
    """

    for inference only

    :param prediction_vectors: [batch_size, num_query_cubes, prediction_vector],
    here prediction vector = image_channel * X * Y * Z, image_channel is 1
    :param list_query_sequence:
    :param cube_shape:
    :return: same shape with list_query_sequence: replace the location to ct_data cube
    """
    batch_size = len(list_query_sequence)
    prediction_arrays = prediction_vectors.cpu().detach().numpy()
    return_list = []
    for i in range(batch_size):
        current_batch_list = []
        for j in range(len(list_query_sequence[i])):
            current_batch_list.append(np.reshape(prediction_arrays[i, j, :], cube_shape))
        return_list.append(current_batch_list)
    return return_list


def post_process_to_dict(prediction_vectors, list_query_sequence, cube_shape, list_mass_center):
    """

    for inference only

    :param prediction_vectors: [batch_size, num_query_cubes, prediction_vector],
    here prediction vector = image_channel * X * Y * Z, image_channel is 1
    :param list_query_sequence:
    :param cube_shape:
    :param list_mass_center: the mass center for each sequence 
    :return: same shape with list_query_sequence: replace the location to dict:
    {'ct_data': ct_cube, 'location_offset': location_offset, 'center_location': center_location}
    """
    batch_size = len(list_query_sequence)
    prediction_arrays = prediction_vectors.cpu().detach().numpy()
    return_list = []
    for i in range(batch_size):
        current_batch_list = []
        mass_center = list_mass_center[i]
        for j in range(len(list_query_sequence[i])):
            ct_cube = np.reshape(prediction_arrays[i, j, :], cube_shape)
            location_offset = list_query_sequence[i][j]
            center_location = (location_offset[0] + mass_center[0], location_offset[1] + mass_center[1], 
                               location_offset[2] + mass_center[2])
            dict_item = {"ct_data": ct_cube, "location_offset": location_offset, "center_location": center_location}
            current_batch_list.append(dict_item)
        return_list.append(current_batch_list)
    return return_list


def post_process_to_tensor(prediction_vectors, cube_shape):
    """

    :param prediction_vectors: [batch_size, num_cubes, prediction_vector],
    here prediction vector = image_channel * X * Y * Z, image_channel is 1
    :param cube_shape:
    :return: [batch_size, num_cubes, 1, X, Y, Z]
    """
    shape_vectors = prediction_vectors.shape
    return torch.reshape(prediction_vectors,
                         (shape_vectors[0], shape_vectors[1], 1, cube_shape[0], cube_shape[1], cube_shape[2]))


def prepare_tensors_3d_mae(list_information_sequence, list_query_sequence, embed_dim, decoding_dim, given_dim=0,
                           device='cuda:0'):
    """
    prepare batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape

    1) list of information sequence: a list of lists of dict, each dict contains {'ct_data': ct_cube,
    'location_offset': center_location_offset, 'given_vector': given_vector}

    list of information sequence = [[dict, dict, dict, ...], [dict, dict, dict, ...], ...], the length is batch size
    data on CPU

    2) list of query sequence: a list of lists of location offsets, like (-10, 12, 13)

    list of query sequence = [[tuple, tuple, tuple, ...], [tuple, tuple, tuple, ...], ...], the length is batch size
    data on CPU

    :return: tensors for model_guided forward
    """
    batch_size = len(list_information_sequence)
    assert batch_size == len(list_query_sequence) and batch_size > 0

    input_sequence_len = 0
    query_sequence_len = 0

    for i in range(batch_size):
        if len(list_information_sequence[i]) > input_sequence_len:
            input_sequence_len = len(list_information_sequence[i])

    for i in range(batch_size):
        if len(list_query_sequence[i]) > query_sequence_len:
            query_sequence_len = len(list_query_sequence[i])

    cube_shape = np.shape(list_information_sequence[0][0]['ct_data'])
    batch_array = np.zeros([batch_size, 1, cube_shape[0], cube_shape[1],
                            cube_shape[2] * input_sequence_len], 'float32')
    location_list = []

    if given_dim > 0:
        given_vector_array = np.zeros([batch_size, input_sequence_len, given_dim], 'float32')
    else:
        given_vector_array = None

    # complete batch_array, source_arrays and given_vector_array
    for i in range(batch_size):
        for j in range(len(list_information_sequence[i])):
            item = list_information_sequence[i][j]
            batch_array[i, 0, :, :, j * cube_shape[2]: (j + 1) * cube_shape[2]] = item['ct_data']
            location_list.append(item['location_offset'])
            if given_dim > 0:
                given_vector_array[i, j, :] = item['given_vector']

    pos_embed_array = get_3d_sincos_pos_embed_loc_list(embed_dim, location_list)
    pos_embed_array_final = np.zeros([batch_size, input_sequence_len, embed_dim], 'float32')
    shift = 0
    for i in range(batch_size):
        for j in range(len(list_information_sequence[i])):
            pos_embed_array_final[i, j, :] = pos_embed_array[shift, :]
            shift += 1

    # prepare input for function "forward_encoder"
    batch_tensor = torch.FloatTensor(batch_array).cuda(device)
    pos_embed_tensor = torch.FloatTensor(pos_embed_array_final).cuda(device)
    if given_dim > 0:
        given_vector = torch.FloatTensor(given_vector_array).cuda(device)
    else:
        given_vector = None

    # prepare query_vectors, which is in [batch_size, self.query_sequence_len, self.decoding_dim]
    location_list_query = []
    for i in range(batch_size):
        for j in range(query_sequence_len):
            if j < len(list_query_sequence[i]):
                location_list_query.append(list_query_sequence[i][j])
            else:
                location_list_query.append((500, 500, 500))  # appended an outlier

    pos_embed_array_query = get_3d_sincos_pos_embed_loc_list(decoding_dim, location_list_query)
    pos_embed_array_query = np.reshape(pos_embed_array_query,
                                       [batch_size, query_sequence_len, decoding_dim])
    query_vector = torch.FloatTensor(pos_embed_array_query).cuda(device)

    return batch_tensor, pos_embed_tensor, given_vector, query_vector, cube_shape


def prepare_tensors_pe_transformer(list_sample_sequence, embed_dim, device='cuda:0', get_flatten_vessel_mask=True,
                                   training_phase=True, guide_depth=4):
    """
    prepare batch_tensor, pos_embed_tensor, given_features, flatten_vessel_mask, cube_shape, clot_gt_tensor

    :param guide_depth: neglect vessels with depth < guide_depth
    :param list_sample_sequence: a list, length is batch_size, each item is a sample sequence. Each sample sequence is
    a list, the item is a dict:
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'branch_level': 'branch_level',
    'clot_array': None}

    :param training_phase: during training, calculate the clot_gt_tensor
    :param get_flatten_vessel_mask:
    :param embed_dim: int
    :param device:
    :return: batch_tensor, pos_embed_tensor, given_features, flatten_vessel_mask, cube_shape
    """
    batch_size = len(list_sample_sequence)
    assert batch_size > 0

    example_sample = list_sample_sequence[0][0]

    sample_sequence_len = 0
    for sample_sequence in list_sample_sequence:
        if len(sample_sequence) > sample_sequence_len:
            sample_sequence_len = len(sample_sequence)

    cube_shape = np.shape(example_sample['ct_data'])
    batch_array = np.zeros([batch_size, 1, cube_shape[0], cube_shape[1],
                            cube_shape[2] * sample_sequence_len], 'float32')
    if example_sample['given_vector'] is None:
        given_vector_array = None
        given_dim = 0
    else:
        given_vector_array = np.zeros(
            [batch_size, sample_sequence_len, len(example_sample['given_vector'])], 'float32')
        given_dim = len(example_sample['given_vector'])

    if get_flatten_vessel_mask:
        flatten_vessel_mask = np.zeros(
            [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')
    else:
        flatten_vessel_mask = None
    if training_phase:
        gt_vectors = np.zeros(
            [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')
    else:
        gt_vectors = None

    location_list = []

    # complete these arrays
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            item = list_sample_sequence[i][j]
            batch_array[i, 0, :, :, j * cube_shape[2]: (j + 1) * cube_shape[2]] = item['ct_data']
            x_c, y_c, z_c = item['location_offset']
            branch_level = item['branch_level']
            location_list.append((x_c, y_c, z_c, branch_level))
            if given_dim > 0:
                given_vector_array[i, j, :] = item['given_vector']
            if get_flatten_vessel_mask:
                flatten_vessel_mask[i, j, :] = \
                    np.reshape(np.array(item['depth_cube'] >= guide_depth, 'float32'), (-1, ))
            if training_phase:
                if item['clot_array'] is not None:  # greater than 0 means this is a simulated clot
                    gt_vectors[i, j, :] = np.reshape(np.array(item['clot_array'] > 0, 'float32'), (-1, ))

    # get positional encoding array
    pos_embed_array_temp = get_4d_sincos_pos_embed_loc_list(embed_dim, location_list)  # [len(loc_list), embed_dim]
    pos_embed_array = np.zeros([batch_size, sample_sequence_len, embed_dim], 'float32')
    shift = 0
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            pos_embed_array[i, j, :] = pos_embed_array_temp[shift, :]
            shift += 1

    # form torch tensors
    batch_tensor = torch.FloatTensor(batch_array).cuda(device)
    pos_embed_tensor = torch.FloatTensor(pos_embed_array).cuda(device)
    if given_dim > 0:
        given_vector = torch.FloatTensor(given_vector_array).cuda(device)
    else:
        given_vector = None

    if get_flatten_vessel_mask:
        flatten_vessel_mask = torch.FloatTensor(flatten_vessel_mask).cuda(device)
    else:
        flatten_vessel_mask = None

    if training_phase:
        gt_vectors_negative = np.array(1 - gt_vectors, 'float32')
        clot_gt_tensor_positive = torch.FloatTensor(gt_vectors).cuda(device)
        clot_gt_tensor_negative = torch.FloatTensor(gt_vectors_negative).cuda(device)

        clot_gt_tensor = torch.stack((clot_gt_tensor_negative, clot_gt_tensor_positive), dim=1)
        # [B, 2, N, flatten_dim]
    else:
        clot_gt_tensor = None

    return batch_tensor, pos_embed_tensor, given_vector, flatten_vessel_mask, cube_shape, clot_gt_tensor


def prepare_tensors_pe_transformer_v3(list_sample_sequence, embed_dim, device='cuda:0', training_phase=True):
    """
    prepare batch_tensor, pos_embed_tensor, given_features, flatten_blood_region, cube_shape, clot_gt_tensor

    :param list_sample_sequence: a list, length is batch_size, each item is a sample sequence. Each sample sequence is
    a list, the item is a dict:
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, 'branch_level': 'branch_level',
    'clot_array': None}

    :param training_phase: during training, calculate the clot_gt_tensor
    :param embed_dim: int
    :param device:
    :return: batch_tensor, pos_embed_tensor, given_features, flatten_vessel_mask, cube_shape
    """
    batch_size = len(list_sample_sequence)
    assert batch_size > 0

    example_sample = list_sample_sequence[0][0]

    sample_sequence_len = 0
    for sample_sequence in list_sample_sequence:
        if len(sample_sequence) > sample_sequence_len:
            sample_sequence_len = len(sample_sequence)

    cube_shape = np.shape(example_sample['ct_data'])
    batch_array = np.zeros([batch_size, 1, cube_shape[0], cube_shape[1],
                            cube_shape[2] * sample_sequence_len], 'float32')
    if example_sample['given_vector'] is None:
        given_vector_array = None
        given_dim = 0
    else:
        given_vector_array = np.zeros(
            [batch_size, sample_sequence_len, len(example_sample['given_vector'])], 'float32')
        given_dim = len(example_sample['given_vector'])

    flatten_blood_region = np.zeros(
        [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')

    if training_phase:
        gt_vectors = np.zeros(
            [batch_size, sample_sequence_len, int(cube_shape[0] * cube_shape[1] * cube_shape[2])], 'float32')
    else:
        gt_vectors = None

    location_list = []

    # complete these arrays
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            item = list_sample_sequence[i][j]
            batch_array[i, 0, :, :, j * cube_shape[2]: (j + 1) * cube_shape[2]] = item['ct_data']
            x_c, y_c, z_c = item['location_offset']
            branch_level = item['branch_level']
            location_list.append((x_c, y_c, z_c, branch_level))
            if given_dim > 0:
                given_vector_array[i, j, :] = item['given_vector']

            blood_region = item["blood_region"] * np.array(
                item['depth_cube'] > max(0.5, 6 - item['branch_level']), 'float32')

            flatten_blood_region[i, j, :] = np.reshape(blood_region, (-1, ))
            if training_phase:
                if item['clot_array'] is not None:  # greater than 0 means this is a simulated clot
                    gt_vectors[i, j, :] = np.reshape(np.array(item['clot_array'] > 0, 'float32'), (-1, ))

    # get positional encoding array
    pos_embed_array_temp = get_4d_sincos_pos_embed_loc_list(embed_dim, location_list)  # [len(loc_list), embed_dim]
    pos_embed_array = np.zeros([batch_size, sample_sequence_len, embed_dim], 'float32')
    shift = 0
    for i in range(batch_size):
        for j in range(len(list_sample_sequence[i])):
            pos_embed_array[i, j, :] = pos_embed_array_temp[shift, :]
            shift += 1

    # form torch tensors
    batch_tensor = torch.FloatTensor(batch_array).cuda(device)
    pos_embed_tensor = torch.FloatTensor(pos_embed_array).cuda(device)
    if given_dim > 0:
        given_vector = torch.FloatTensor(given_vector_array).cuda(device)
    else:
        given_vector = None

    flatten_blood_region = torch.FloatTensor(flatten_blood_region).cuda(device)

    if training_phase:
        gt_vectors_negative = np.array(1 - gt_vectors, 'float32')
        clot_gt_tensor_positive = torch.FloatTensor(gt_vectors).cuda(device)
        clot_gt_tensor_negative = torch.FloatTensor(gt_vectors_negative).cuda(device)

        clot_gt_tensor = torch.stack((clot_gt_tensor_negative, clot_gt_tensor_positive), dim=1)
        # [B, 2, N, flatten_dim]
    else:
        clot_gt_tensor = None

    return batch_tensor, pos_embed_tensor, given_vector, flatten_blood_region, cube_shape, clot_gt_tensor


def get_list_prediction_sequence(list_sample_sequence, segment_probability, key='clot_array'):
    """

    :param list_sample_sequence:
    :param segment_probability: in numpy float32 [batch_size, 2, N, flatten_dim],
                                    for second channel, 0 for negative, 1 for positive
    :param key: assign the positive channel to this key
    :return:
    [[{'location_offset': central_location_offset, 'center_location': central_location, key: predicted_array}, ..], ..]
    """
    batch_size = len(list_sample_sequence)

    cube_shape = np.shape(list_sample_sequence[0][0]['ct_data'])  # like (5, 5, 5)

    assert cube_shape[0] * cube_shape[1] * cube_shape[2] == np.shape(segment_probability)[3]  # equals flatten_dim

    list_prediction_sequence = []
    for i in range(batch_size):
        sample_sequence = list_sample_sequence[i]
        prediction_sequence = []
        for j in range(len(sample_sequence)):
            item = sample_sequence[j]

            new_item_dict = {'location_offset': item['location_offset'], 'center_location': item['center_location']}
            predict_mask = np.reshape(segment_probability[i, 1, j, :], cube_shape)
            new_item_dict[key] = predict_mask

            prediction_sequence.append(new_item_dict)

        list_prediction_sequence.append(prediction_sequence)

    return list_prediction_sequence


def form_flatten_mask_mae(batch_sample, device='cuda:0'):
    """

    compared to "form_tensors_tissue_wise", this version require the model_guided to predict information cubes

    :param batch_sample: {"list_sample_sequence":, "list_query_gt_sequence":},
    is the iteration output of the "DataLoaderForPEIte"
    :param device

    :return: four tensors on GPU,
    batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query
    in shape
    [batch_size, input_channel, X, Y, Z * information_sequence_len],
    [batch_size, input_channel, X, Y, Z * query_sequence_len],
    [batch_size, input_channel, X, Y, Z * information_sequence_len],
    [batch_size, input_channel, X, Y, Z * query_sequence_len],

    """

    list_information_sequence = batch_sample["list_sample_sequence"]
    list_query_gt_sequence = batch_sample["list_query_gt_sequence"]
    """
    each item in the above list:
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
            'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube}
    """

    batch_size = len(list_information_sequence)
    num_query_cubes = 0
    num_information_cubes = 0

    for i in range(batch_size):
        if len(list_query_gt_sequence[i]) > num_query_cubes:
            num_query_cubes = len(list_query_gt_sequence[i])
        if len(list_information_sequence[i]) > num_information_cubes:
            num_information_cubes = len(list_information_sequence[i])

    x, y, z = np.shape(list_query_gt_sequence[0][0]['depth_cube'])

    depth_array_info = np.zeros([batch_size, 1, x, y, z * num_information_cubes], 'float32')
    depth_array_query = np.zeros([batch_size, 1, x, y, z * num_query_cubes], 'float32')

    for i in range(batch_size):
        for j in range(len(list_information_sequence[i])):  # form the encoding_depth mask for information part
            item = list_information_sequence[i][j]
            depth_array_info[i, 0, :, :, j * z: (j + 1) * z] = item['depth_cube']

        for j in range(len(list_query_gt_sequence[i])):  # form the encoding_depth mask for query part
            item = list_query_gt_sequence[i][j]
            depth_array_query[i, 0, :, :, j * z: (j + 1) * z] = item['depth_cube']

    mask_array_info = np.array(depth_array_info > 0.5, 'float32')
    mask_array_query = np.array(depth_array_query > 0.5, 'float32')

    batch_depth_info = torch.FloatTensor(depth_array_info).cuda(device)
    batch_depth_query = torch.FloatTensor(depth_array_query).cuda(device)
    batch_mask_info = torch.FloatTensor(mask_array_info).cuda(device)
    batch_mask_query = torch.FloatTensor(mask_array_query).cuda(device)

    return batch_mask_info, batch_mask_query, batch_depth_info, batch_depth_query


class OutlierLossDetect:
    def __init__(self, store_count=30, remove_max_count=3, remove_min_count=3, std_outlier=10):
        self.recent_loss_history = []
        self.store_count = store_count
        self.remove_max_count = remove_max_count
        self.remove_min_count = remove_min_count
        self.std_outlier = std_outlier

        self.in_queue_count = 0

        self.consecutive_outlier = 0

    def update_new_loss(self, new_loss):
        # return True for non-outlier loss

        if self.consecutive_outlier > 20:
            print("consecutive outlier detected!")
            return "consecutive_outlier"

        if len(self.recent_loss_history) < self.store_count:
            self.recent_loss_history.append(new_loss)
            self.in_queue_count += 1
            self.consecutive_outlier = 0
            return True

        std_in_queue, ave_in_queue = self.get_std_and_ave_in_queue()
        lower_bound = ave_in_queue - self.std_outlier * std_in_queue
        upper_bound = ave_in_queue + self.std_outlier * std_in_queue

        if new_loss < lower_bound or new_loss > upper_bound:
            print("outlier loss:", new_loss)
            print("average recent", len(self.recent_loss_history),
                  "loss:", ave_in_queue, "std for recent", len(self.recent_loss_history), "loss:", std_in_queue)
            self.consecutive_outlier += 1
            return False

        self.recent_loss_history[self.in_queue_count % self.store_count] = new_loss
        self.in_queue_count += 1
        self.consecutive_outlier = 0
        return True

    def reset(self):
        self.in_queue_count = 0
        self.recent_loss_history = []
        self.consecutive_outlier = 0

    def get_std_and_ave_in_queue(self):

        if len(self.recent_loss_history) < self.store_count:
            std_in_queue = np.std(self.recent_loss_history)
            ave_in_queue = np.average(self.recent_loss_history)
            return std_in_queue, ave_in_queue

        temp_list = list(self.recent_loss_history)
        temp_list.sort()
        std_in_queue = np.std(temp_list[self.remove_min_count: -self.remove_max_count])
        ave_in_queue = np.average(temp_list[self.remove_min_count: -self.remove_max_count])

        return std_in_queue, ave_in_queue


class TrainingPhaseControl:
    def __init__(self, params):

        self.target_recall = params["target_recall"]
        self.target_precision = params["target_precision"]

        self.flip_recall = params["flip_recall"]
        self.flip_precision = params["flip_precision"]

        self.base_recall = params["base_recall"]
        self.base_precision = params["base_precision"]

        self.current_phase = 'warm_up'
        # 'warm_up', 'recall_phase', 'precision_phase', 'converge_to_recall', 'converge_to_precision'

        self.flip_remaining = params["flip_remaining"]
        # one flip means change the phase 'precision_phase' -> 'recall_phase'

        self.base_relative = params["base_relative"]
        # will not flip util number times recall/precision bigger than precision/recall >= base_relative

        self.max_performance_recall = params["max_performance_recall"]
        self.max_performance_precision = params["max_performance_precision"]
        # force flip when precision/recall > max_performance during precision/recall phase

        self.final_phase = params["final_phase"]  # 'converge_to_recall', 'converge_to_precision'

        self.warm_up_epochs = params["warm_up_epochs"]

        self.previous_phase = None
        self.changed_phase_in_last_epoch = False

        # --------------------------
        # check correctness
        assert 0 <= self.flip_recall <= 1 and 0 <= self.flip_precision <= 1
        assert 0 <= self.base_recall <= 1 and 0 <= self.base_precision <= 1
        assert self.flip_remaining >= 0
        assert self.warm_up_epochs >= 0

        assert self.final_phase in ['converge_to_recall', 'converge_to_precision']
        if self.final_phase == 'converge_to_recall':
            assert 0 < self.target_recall < 1
        if self.final_phase == 'converge_to_precision':
            assert 0 < self.target_precision < 1

        self.precision_to_recall_during_converging = 4
        # the precision and recall will fluctuate around the target performance. When this value to 0, end to training.

        self.epoch_passed = 0
        self.relative_false_positive_penalty = params["initial_relative_false_positive_penalty"]
        # higher means model give less false positives, at the expense of more false negative

        self.history_relative_false_positive_penalty = []
        self.history_recall = []
        self.history_precision = []

    def get_new_relative_false_positive_penalty(self, current_recall, current_precision):
        self._update_history(current_recall, current_precision)
        self.changed_phase_in_last_epoch = self._update_phase(current_recall, current_precision)
        self._update_relative_false_positive_penalty(current_recall, current_precision)
        self.show_status(current_recall, current_precision)
        self.epoch_passed += 1
        return self.relative_false_positive_penalty

    def _update_history(self, current_recall, current_precision):
        self.history_relative_false_positive_penalty.append(self.relative_false_positive_penalty)
        self.history_recall.append(current_recall)
        self.history_precision.append(current_precision)

    def _update_phase(self, current_recall, current_precision):
        # return True for phase change

        if self.previous_phase is None:
            self.previous_phase = self.current_phase  # update previous phase when update current phase

        if self.current_phase == self.final_phase:  # do not update
            return False

        if self.epoch_passed < self.warm_up_epochs:
            self.current_phase = 'warm_up'
            return False

        if self.current_phase == 'warm_up' and self.epoch_passed >= self.warm_up_epochs:
            self.current_phase = 'recall_phase'
            if (current_recall > self.flip_recall and current_recall / (current_precision + 1e-8) > self.base_relative)\
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                self.previous_phase = self.current_phase
                self.current_phase = 'precision_phase'
            print("changing current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
            return True

        if self.current_phase == 'recall_phase':
            if (current_recall > self.flip_recall and current_recall / (current_precision + 1e-8) > self.base_relative)\
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                if self.flip_remaining > 0 or self.final_phase == 'converge_to_precision':
                    self.previous_phase = self.current_phase
                    self.current_phase = 'precision_phase'
                else:
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                print("change current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
                return True

        if self.current_phase == 'precision_phase':
            if (current_precision > self.flip_precision
                and current_precision / (current_recall + 1e-8) > self.base_relative) \
                    or current_recall < self.base_recall or current_precision > self.max_performance_precision:
                if self.flip_remaining > 0:
                    self.previous_phase = self.current_phase
                    self.current_phase = 'recall_phase'
                    self.flip_remaining -= 1
                    print("changing current_phase to:", self.current_phase, 'flip_remaining', self.flip_remaining)
                    return True
                else:
                    assert self.final_phase == 'converge_to_precision'
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                    print("change current_phase to:", self.current_phase)
                    return True
        return False

    def show_status(self, current_recall=None, current_precision=None):
        print("epoch passed:", self.epoch_passed, "current phase:", self.current_phase,
              "relative_false_positive_penalty", self.relative_false_positive_penalty,
              "flip remaining:", self.flip_remaining)
        if current_recall is not None and current_precision is not None:
            print("current (recall, precision)", (current_recall, current_precision))

    def _update_relative_false_positive_penalty(self, current_recall, current_precision):

        if self.current_phase == 'warm_up':
            print("warm_up phase, relative_false_positive_penalty:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'recall_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.15
            print("recall phase, decrease relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'precision_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.13
            print("precision phase, increase relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_recall':

            if current_recall > self.target_recall:  # the recall is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_precision':

            if current_precision > self.target_precision:  # the precision is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty


if __name__ == '__main__':
    exit()
