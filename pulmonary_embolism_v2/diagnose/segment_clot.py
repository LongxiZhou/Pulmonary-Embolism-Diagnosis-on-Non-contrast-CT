"""
the abnormal detection is based on the difference between expected and ground truth.

see function "get_expected_ct_signal" to get the expected ct

"""
from pulmonary_embolism.model_pretraining.convert_blood_vessel_to_sliced_sequence import convert_ct_into_tubes
import med_transformer.image_transformer.transformer_for_PE.predict_vessel_sequence as predict_sequence
import basic_tissue_prediction.predict_rescaled as predict_rescaled
import format_convert.spatial_normalize as spatial_normalize
import analysis.center_line_and_depth_3D as center_line_and_depth
import format_convert.dcm_np_converter_new as data_extract
import Tool_Functions.Functions as Functions
import pulmonary_embolism.sequence_operations.trim_length as trim_sequence
import numpy as np
import random
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def prepare_data(dcm_dict_or_rescaled_array, blood_vessel_mask=None, checkpoint_path_upsample=None, show=True,
                 batch_size=8, refine_vessel=True, return_center_line=False):
    """
    :param return_center_line:
    :param dcm_dict_or_rescaled_array:
    :param blood_vessel_mask:
    :param checkpoint_path_upsample:
    :param batch_size: batch_size when upsample or predict blood vessel.
    :param show:
    :param refine_vessel:
    :return: the rescaled_ct, the mask for blood vessel, the depth_array for blood vessel and the center line
    """
    if type(dcm_dict_or_rescaled_array) is str:
        rescaled_ct = data_extract.establish_rescale_chest_ct(dcm_dict_or_rescaled_array,
                                                              checkpoint_path_upsample, show, batch_size)
    else:
        rescaled_ct = dcm_dict_or_rescaled_array

    if blood_vessel_mask is None:
        blood_vessel_mask = predict_rescaled.get_prediction_blood_vessel(rescaled_ct, refine_blood_vessel=refine_vessel,
                                                                         batch_size=batch_size)
    depth_mask = center_line_and_depth.get_surface_distance(blood_vessel_mask)

    if return_center_line:
        center_line_mask = center_line_and_depth.get_center_line(blood_vessel_mask, surface_distance=depth_mask)
    else:
        center_line_mask = None

    return rescaled_ct, blood_vessel_mask, depth_mask, center_line_mask


def extract_multiple_sequence_for_vessel_region(rescaled_ct, blood_vessel_mask, depth_mask, shift_list=None,
                                                high_resolution=False):
    """
    we will extract several sequence for one scan, the difference between sequence is the "shift", which makes the
    location of each voxel in a cube be uniform: each voxel will near the corner of a cube in one sequence, but will
    near the center in other sequence.

    :param high_resolution:
    :param rescaled_ct:
    :param blood_vessel_mask:
    :param depth_mask:
    :param shift_list: item is the shift for each sequence
    :return: list of sequences. each sequence corresponding the the "shift" in the shift_list
    """
    if shift_list is None:
        # the cube_shape extracted from (512, 512, 512) is (11, 11, 11), i.e., cube around 7mm by 7mm by 10mm
        shift_list = generate_shift_list()  # 27 shift tuples

    # calculate the mass center
    min_depth = 4  # we remove encoding_depth < 4 when calculate the mass center
    location_array = np.where(depth_mask > (min_depth - 0.5))
    mass_center = (int(np.average(location_array[0])), int(np.average(location_array[1])),
                   int(np.average(location_array[2])))
    print("mass center:", mass_center)

    # calculate the bounding box for blood vessels
    cube_length = (11, 11, 11)
    bounding_box = Functions.get_bounding_box(blood_vessel_mask, pad=int(min(cube_length) / 2))
    print("bounding box for blood vessel is:", bounding_box)

    sequence_list = []
    for shift in shift_list:
        if not high_resolution:
            sequence = convert_ct_into_tubes(rescaled_ct, blood_vessel_mask, min_depth=min_depth, show=False,
                                             mass_center=mass_center, depth_array=depth_mask, only_v1=True,
                                             shift=shift, absolute_cube_length=(7, 7, 10))
        else:
            sequence = convert_ct_into_tubes(rescaled_ct, blood_vessel_mask, min_depth=min_depth, show=False,
                                             mass_center=mass_center, depth_array=depth_mask, only_v1=True,
                                             shift=shift, absolute_cube_length=(4, 4, 5))
            if len(sequence) > 3000:
                print("original length", len(sequence), "trim to 3000")
            sequence = trim_sequence.reduce_sequence_length(sequence)
        print("sequence length:", len(sequence))
        sequence_list.append(sequence)

    return sequence_list


def iterative_predict_sequence_list(sequence_list, mask_ratio=0.5, ergodic_count=4, batch_size=2, model=None,
                                    model_path=None, show=True, high_resolution=False):
    """

    :param high_resolution:
    :param sequence_list: sequence of a ct scan, return of "extract_multiple_sequence_for_vessel_region"
    :param mask_ratio: when predict the rest cubes, use 1 - mask_ratio of cubes.

    ergodic_count / (1 - mask_ratio) should be a int!

    :param ergodic_count: number of times each cube be predicted
    :param batch_size: usually equals to your GPU number

    :param model: the model_guided for predict
    :param model_path: path to load the model_guided
    :param show
    :return: prediction_cube_list, a list, each item is a dict in
     {"ct_data": ct_cube, "location_offset": location_offset, "center_location": center_location}
    """
    assert round(ergodic_count / (1 - mask_ratio)) == round(ergodic_count / (1 - mask_ratio), 10)

    assert type(ergodic_count) is int

    fold_count = int(1 / (1 - mask_ratio))
    if show:
        print("each sequence will be cross predict", fold_count * ergodic_count, 'times')

    if model_path is not None:
        assert model is None
    if model is None:
        print("loading model_guided...")
        model = predict_sequence.load_saved_model_guided(model_path, high_resolution=high_resolution)

    prediction_cube_list = []

    information_query_pair_list = []  # each item is (information_sequence, query_sequence)
    # calculate the information_query_pair_list
    for sequence in sequence_list:

        length_information = int(len(sequence) * (1 - mask_ratio))

        for epoch in range(ergodic_count):
            random.shuffle(sequence)
            for fold in range(fold_count):
                # establish information sequence and query sequence
                info_index_start = int(len(sequence) / fold_count * fold)
                info_index_end = min(len(sequence), info_index_start + length_information)

                cross_end = False
                info_sequence = sequence[info_index_start: info_index_end]
                current_info_len = len(info_sequence)
                if current_info_len < length_information:
                    info_sequence = info_sequence + sequence[0: length_information - current_info_len]
                    info_index_end = length_information - current_info_len
                    cross_end = True

                query_gt_sequence = []

                if cross_end is False:
                    for index in range(0, info_index_start):
                        query_gt_sequence.append(sequence[index])
                    for index in range(info_index_end, len(sequence)):
                        query_gt_sequence.append(sequence[index])
                else:
                    for index in range(info_index_end, info_index_start):
                        query_gt_sequence.append(sequence[index])

                assert len(query_gt_sequence) + len(info_sequence) == len(sequence)

                information_query_pair_list.append((info_sequence, query_gt_sequence))

    # predicting these pairs
    print(len(information_query_pair_list), "number of predicting tasks.")
    for index in range(0, len(information_query_pair_list), batch_size):
        index_start = index
        index_end = min(index + batch_size, len(information_query_pair_list))

        list_information_sequence = []
        list_query_sequence = []

        for sample_count in range(index_end - index_start):
            list_information_sequence.append(information_query_pair_list[index_start + sample_count][0])
            list_query_sequence.append(information_query_pair_list[index_start + sample_count][1])

        list_prediction = predict_sequence.predict_cube_sequence(list_information_sequence, list_query_sequence, model,
                                                                 high_resolution=high_resolution)
        # list_prediction is like [[dict, dict, ...], [dict, dict, ...], ...]
        for sequence_predicted in list_prediction:
            prediction_cube_list = prediction_cube_list + sequence_predicted

    return prediction_cube_list


def reconstruct_expected(prediction_cube_list, return_predict_count_mask, cube_length, detailed_history=False):
    """

    :param detailed_history:
    :param cube_length:
    :param prediction_cube_list: a list of dict, each dict like
    {"ct_data": ct_cube, "location_offset": location_offset, "center_location": center_location}
    :param return_predict_count_mask, if True, return all_file prediction
    :return:
    the expected array in shape [512, 512, 512]. If a voxel is predicted multiple times, use the average.
    the prediction_history, is a dict, key is the voxel location like "(234, 256, 180)", content is the list of
    predicted values like [prediction_1, prediction_2, ...]
    """
    reconstructed_array = np.zeros([512, 512, 512], 'float32')
    predict_count_array = np.zeros([512, 512, 512], 'int16')

    cube_radius_x = int(cube_length[0] / 2)
    cube_radius_y = int(cube_length[1] / 2)
    cube_radius_z = int(cube_length[2] / 2)

    prediction_history = {}
    voxel_location_key_set = set()

    def update_prediction_history(cube_center_location, predicted_cube):
        center_x, center_y, center_z = cube_center_location
        for shift_x in range(-cube_radius_x, cube_radius_x + 1):
            for shift_y in range(-cube_radius_x, cube_radius_x + 1):
                for shift_z in range(-cube_radius_x, cube_radius_x + 1):
                    voxel_location = (center_x + shift_x, center_y + shift_y, center_z + shift_z)
                    predicted_value = \
                        predicted_cube[-cube_radius_x + shift_x, -cube_radius_y + shift_y, -cube_radius_z + shift_z]
                    location_key = str(voxel_location)
                    if location_key not in voxel_location_key_set:
                        prediction_history[location_key] = [predicted_value]
                    else:
                        prediction_history[location_key].append(predicted_value)

    for item in prediction_cube_list:
        ct_cube = item["ct_data"]  # in shape target_shape, like (5, 5, 5)
        ct_cube = spatial_normalize.rescale_to_new_shape(ct_cube, cube_length)

        center_location = item["center_location"]

        # predict count add one
        predict_count_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                            center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                            center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] = \
            predict_count_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                                center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                                center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] + 1

        # add the ct_tube to the reconstruct array
        reconstructed_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                            center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                            center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] = \
            reconstructed_array[center_location[0] - cube_radius_x: center_location[0] + cube_radius_x + 1,
                                center_location[1] - cube_radius_y: center_location[1] + cube_radius_y + 1,
                                center_location[2] - cube_radius_z: center_location[2] + cube_radius_z + 1] + ct_cube

        # update history
        if detailed_history:
            update_prediction_history(center_location, ct_cube)

    reconstructed_array = reconstructed_array / np.clip(predict_count_array, 1, np.inf)

    if return_predict_count_mask and detailed_history:
        return reconstructed_array, predict_count_array, prediction_history

    if return_predict_count_mask:
        return reconstructed_array, predict_count_array

    return reconstructed_array


def generate_shift_list(axis_shift_x=(-1, 0, 1), axis_shift_y=(-1, 0, 1), axis_shift_z=(-1, 0, 1)):
    shift_list = []
    for shift_x in axis_shift_x:
        for shift_y in axis_shift_y:
            for shift_z in axis_shift_z:
                shift_list.append((shift_x, shift_y, shift_z))
    return shift_list


def get_expected_ct_signal(dcm_dict_or_rescaled_array, blood_vessel_mask=None, checkpoint_path_upsample=None, show=True,
                           batch_size_upsample=8,
                           refine_vessel=True, shift_list=None, mask_ratio=0.5, ergodic_count=4, batch_size_mae=2,
                           model_guided=None, model_guided_path=None, return_detailed_history=False,
                           return_ct_data_package=False, high_resolution=False):
    """

    :param high_resolution:
    :param dcm_dict_or_rescaled_array:
    :param blood_vessel_mask:
    :param checkpoint_path_upsample:
    :param show:
    :param batch_size_upsample:
    :param refine_vessel:
    :param shift_list: the list of shift when convert the ct array into sequence
    :param mask_ratio:
    :param ergodic_count:
    :param batch_size_mae:
    :param model_guided:
    :param model_guided_path:

    :param return_detailed_history:
    :param return_ct_data_package:
    :return:
    expected_array in numpy [512, 512, 512] float32, prediction_history (optional), ct_data_package (optional)

    about prediction_history: dict, key is the location like "(234, 345, 256)", value is list of each predicted value.
    about the ct_data_package, is the return of function "prepare_data", i.e. a tuple of four elements:
    (rescaled_ct, blood_vessel_depth, depth_array, center_line_mask)
    each item is a [512, 512, 512] numpy float32 array
    """

    if shift_list is None:
        # the cube_shape extracted from (512, 512, 512) is (11, 11, 11), i.e., cube around 7mm by 7mm by 10mm
        shift_list = generate_shift_list()  # 27 shift tuples

    # get the data package
    if show:
        print("extract data package")
    rescaled_ct, blood_vessel_mask, depth_mask, center_line_mask = \
        prepare_data(dcm_dict_or_rescaled_array, blood_vessel_mask, checkpoint_path_upsample, show,
                     batch_size_upsample, refine_vessel)

    # each scan will be convert to sequence with different shift
    if show:
        print("extract sequence list")
    sequence_list = extract_multiple_sequence_for_vessel_region(rescaled_ct, blood_vessel_mask, depth_mask, shift_list,
                                                                high_resolution=high_resolution)

    # the prediction of mae
    if show:
        print("predict sequence list")
    prediction_cube_list = iterative_predict_sequence_list(sequence_list, mask_ratio, ergodic_count, batch_size_mae,
                                                           model_guided, model_guided_path, show,
                                                           high_resolution=high_resolution)
    # reconstruct the prediction and return
    if show:
        print("reconstruction")

    if high_resolution:
        cube_length = (7, 7, 5)  # refers to (4mm, 4mm, 5mm)
    else:
        cube_length = (11, 11, 11)  # refers to (7mm, 7mm, 10mm)

    if return_detailed_history:
        expected_ct, predict_count_array, prediction_history = \
            reconstruct_expected(prediction_cube_list, True, cube_length, detailed_history=return_detailed_history)

        if return_ct_data_package:
            return expected_ct, prediction_history, (rescaled_ct, blood_vessel_mask, depth_mask, center_line_mask)
        else:
            return expected_ct, prediction_history
    else:
        expected_ct, predict_count_array = reconstruct_expected(
            prediction_cube_list, True, cube_length, detailed_history=return_detailed_history)

        if return_ct_data_package:
            return expected_ct, predict_count_array, (rescaled_ct, blood_vessel_mask, depth_mask, center_line_mask)
        else:
            return expected_ct


def visualize_effect_on_test_set(high_resolution, denoise, fold=(0, 1), name_list=None):
    import collaborators_package.denoise_chest_ct.denoise_predict as de_noising
    import pulmonary_embolism.diagnose.get_radiomics as get_radiomics

    image_save_top_dict = '/home/zhoul0a/Desktop/pulmonary_embolism/visualization/evaluate/temp/' \
                          'high_reso_denoise_old_overlap_mask0.95/'

    dict_for_non_contrast_ct = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/non-contrast/'
    top_dict_semantic = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_masks/non-contrast/'

    model_path = \
        '/home/zhoul0a/Desktop/pulmonary_embolism/check_point_guide/training/high_resolution_denoise_include_rad/best_model_guided.pth'

    model_transformer = predict_sequence.load_saved_model_guided(model_path, high_resolution=high_resolution)

    if denoise:
        model_denoise = de_noising.load_model()
    else:
        model_denoise = None

    if name_list is None:
        name_list = os.listdir(dict_for_non_contrast_ct)[fold[0]::fold[1]]

    processed_count = 0

    for name in name_list:

        print('processing:', name, processed_count, '/', len(name_list))

        if os.path.exists(os.path.join(image_save_top_dict, name, str(499) + '.png')):
            print("processed")
            processed_count += 1
            continue

        if name[-1] == 'z':
            rescaled_ct = np.load(os.path.join(dict_for_non_contrast_ct, name))['array']
        else:
            rescaled_ct = np.load(os.path.join(dict_for_non_contrast_ct, name))
        if denoise:
            rescaled_ct = de_noising.denoise_rescaled_array(rescaled_ct, model_or_model_path=model_denoise)

        blood_vessel_mask = np.load(os.path.join(top_dict_semantic, 'blood_mask/' + name[:-1] + 'z'))['array']

        depth_array = center_line_and_depth.get_surface_distance(blood_vessel_mask)

        inherent_noise = get_radiomics.get_inherent_noise(rescaled_ct, depth_array, is_depth_mask=True, show=True,
                                                          denoise=True)

        expected = get_expected_ct_signal(rescaled_ct, blood_vessel_mask=blood_vessel_mask,
                                          model_guided=model_transformer, high_resolution=high_resolution)

        dep_mask = np.array(depth_array >= 4, 'float32')

        bounding_box_z = Functions.get_bounding_box(dep_mask, pad=2)[2]

        greater = rescaled_ct - expected

        greater = greater * dep_mask * 1600 / (inherent_noise["inherent_noise (denoise)"] + 16)

        greater = np.clip(greater, 0, 1)

        array_rescaled = np.clip(rescaled_ct + 0.5, 0, 1)

        for i in range(bounding_box_z[0], bounding_box_z[1], 1):
            image = Functions.merge_image_with_mask(array_rescaled[:, :, i], greater[:, :, i], show=False)

            Functions.image_save(image, os.path.join(image_save_top_dict, name, str(i) + '.png'), dpi=300)

        processed_count += 1


if __name__ == '__main__':
    visualize_effect_on_test_set(high_resolution=True, denoise=False, name_list=['patient-id-050.npy'])

    exit()
    ct_array = np.load('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/healthy_people/xwzc/xwzc000014.npz')['array']

    import collaborators_package.denoise_chest_ct.denoise_predict as de_noising

    ct_array = de_noising.denoise_rescaled_array(ct_array)

    blood_mask = np.load('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/semantics/healthy_people/xwzc/blood_mask/xwzc000014.npz')['array']
    model_pe_path = '/home/zhoul0a/Desktop/pulmonary_embolism/check_point_guide/training/high_resolution_denoise_include_rad/best_model_guided.pth'
    expect_ct, prediction_count_array, package = get_expected_ct_signal(ct_array, high_resolution=True, model_guided_path=model_pe_path,
                                                       blood_vessel_mask=blood_mask, return_ct_data_package=True)

    print(np.max(package[2]))
    depth_greater_4 = np.array(package[2] >= (np.max(package[2] - 3)), 'float32')
    prediction_count_array = prediction_count_array * depth_greater_4

    loc_list_108 = Functions.get_location_list(np.where(prediction_count_array == 108))

    import Tool_Functions.statistical_tests as tests

    sample_dif_list = []

    for loc in loc_list_108:
        value_dif = ct_array[loc] - expect_ct[loc]
        sample_dif_list.append(value_dif)

    print('there are', len(sample_dif_list))

    print(np.mean(sample_dif_list) * 1600, np.std(sample_dif_list) * 1600, np.mean(np.abs(sample_dif_list) * 1600))

    tests.normality_test(sample_dif_list, show_qq_norm=True, save_path='/home/zhoul0a/Desktop/pulmonary_embolism/figures/normality_for_local_error.svg')

    exit()

    valid_mask = np.array(prediction_count_array == np.max(prediction_count_array), 'float32')



    exit()

    print(Functions.array_stat(prediction_count_array))
    for i in range(200, 350, 10):
        Functions.image_show(expect_ct[:, :, i])
        Functions.image_show(prediction_count_array[:, :, i])

    exit()
    visualize_effect_on_test_set(high_resolution=True, denoise=True, fold=(0, 2))
    exit()
