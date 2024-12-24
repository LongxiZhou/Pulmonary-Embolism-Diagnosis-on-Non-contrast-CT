import Tool_Functions.Functions as Functions
import numpy as np
import os


def reduce_sequence_length(sample_sequence, func_exclusion=None, target_length=3000, max_branch=7):
    """

    :param max_branch: remove a cube if it exceed max_branch
    :param sample_sequence: a list of samples
    :param func_exclusion: input a sample, output a float, higher means first to exclude
    :param target_length:
    :return: sample_sequence_copy of target_length
    """

    if func_exclusion is None:
        func_exclusion = exclusion_large_branch
    original_len = len(sample_sequence)

    new_sample_sequence = []
    for sample in sample_sequence:
        if sample['branch_level'] >= max_branch:
            continue
        else:
            new_sample_sequence.append(sample)

    sample_sequence = new_sample_sequence
    if len(sample_sequence) < target_length:
        final_length = len(sample_sequence)
        print("original length:", original_len, ";  final length:", final_length)
        return sample_sequence

    remove_number = len(sample_sequence) - target_length

    index_exclusion_list = []

    for index in range(len(sample_sequence)):
        sample = sample_sequence[index]
        index_exclusion_list.append((index, func_exclusion(sample)))

    def func_compare(item_1, item_2):
        if item_1[1] > item_2[1]:
            return 1
        return -1

    index_exclusion_list = Functions.customized_sort(index_exclusion_list, func_compare, True)

    list_remove_index = []
    for i in range(remove_number):
        list_remove_index.append(index_exclusion_list[i][0])

    new_sample_sequence = []
    for index in range(len(sample_sequence)):
        if index in list_remove_index:
            continue
        if sample_sequence[index]['branch_level'] >= max_branch:
            continue
        new_sample_sequence.append(sample_sequence[index])

    final_length = len(new_sample_sequence)
    print("original length:", original_len, ";  final length:", final_length)
    return new_sample_sequence


def exclusion_small_vessel(sample):
    depth_cube = sample['depth_cube']
    location_offset = sample['location_offset']
    return np.sum(np.abs(location_offset)) / (np.max(depth_cube) + 1)


def exclusion_large_branch(sample):
    return sample['branch_level']


def trim_dataset(dict_sample_sequence, save_dict_trim, func_exclusion=None, target_length=3000):
    if func_exclusion is None:
        func_exclusion = exclusion_large_branch

    fn_list = os.listdir(dict_sample_sequence)
    count = 0
    for fn in fn_list:

        if count % 10 == 0:
            print(count, '/', len(fn_list))

        sample_sequence = Functions.pickle_load_object(os.path.join(dict_sample_sequence, fn))

        trimmed_sequence = reduce_sequence_length(sample_sequence, func_exclusion, target_length)

        Functions.pickle_save_object(os.path.join(save_dict_trim, fn), trimmed_sequence)

        count += 1


if __name__ == '__main__':

    trim_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/'
                 'temp_dataset/',
                 '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/training_dataset/'
                 'temp_dataset_trim/')
    exit()
    trim_dataset('/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/'
                 'training_dataset/merged-3000/',
                 '/home/zhoul0a/Desktop/pulmonary_embolism/sample_sequence_dataset/simulate_clot/'
                 'training_dataset/merged-refine_length-3000_branch-7/')
    exit()

    # the following code visualize the trim effect
    import med_transformer.image_transformer.transformer_for_3D.rescaled_ct_sample_sequence_converter as reconstruct

    long_sequence = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/pulmonary_embolism/vessel_dataset_high_resolution/merged_v1/disk19-8_2020-04-29.pickle')
    print(len(long_sequence))
    short_sequence = reduce_sequence_length(long_sequence, exclusion_small_vessel, target_length=3000)

    long_sequence_ct = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(long_sequence, (4, 4, 5))
    short_sequence_ct = reconstruct.reconstruct_rescaled_ct_from_sample_sequence(short_sequence, (4, 4, 5))

    for slice_id in range(150, 400, 20):
        Functions.image_show(
            np.concatenate((long_sequence_ct[:, :, slice_id], short_sequence_ct[:, :, slice_id]), axis=1))
