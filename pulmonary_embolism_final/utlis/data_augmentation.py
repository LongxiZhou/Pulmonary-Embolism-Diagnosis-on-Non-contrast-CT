"""
random rotate, flip and swap axis for sequence sample
"""
import Tool_Functions.Functions as Functions
from classic_models.Unet_3D.utlis import random_flip_rotate_swap, get_labels
import numpy as np
import copy


def random_flip_rotate_swap_sample(original_sample, labels=None, reverse=False):
    """
    original sample
    {"center_line_loc_array": , "sample_sequence": , "additional_info": ,}
    the inference only need "sample_sequence", which is a list of dict
    each dict in "sample_sequence":  (cube in float16)
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube, "clot_id_array", None,
    'branch_level': float(branch_level_average), 'clot_array': None, "blood_region": blood_cube}

    :param reverse: undo the augmentation
    augmented_sample = random_flip_rotate_swap_sample(original_sample, labels, reverse=False)
    original_sample = random_flip_rotate_swap_sample(augmented_sample, labels, reverse=True)

    :param labels:
    :param original_sample:
    :return: augmented sample, same structure with original sample
    """
    original_sample = copy.deepcopy(original_sample)

    new_sample = {}
    key_list = list(original_sample.keys())
    for key in key_list:
        if key not in ['sample_sequence', 'center_line_loc_array']:
            new_sample[key] = original_sample[key]

    if labels is None:
        labels = get_labels(swap_axis=False)  # label_flip, label_rotate, label_swap
    else:
        if labels[2] > 0:
            print("original labels:", labels)
            print("remove swap label")
            labels = (labels[0], labels[1], 0)

    # get new center line
    if "center_line_loc_array" in original_sample.keys():
        center_line_loc_array = original_sample["center_line_loc_array"]
        center_line_mask_array = np.zeros([512, 512, 512], 'float32')
        center_line_mask_array[center_line_loc_array] = 1
        new_center_line_mask = random_flip_rotate_swap(center_line_mask_array, False, labels, reverse=reverse)
        new_center_line_loc_array = np.where(new_center_line_mask > 0.5)
        new_sample["center_line_loc_array"] = new_center_line_loc_array

    # get new sample_sequence
    sample_sequence = original_sample["sample_sequence"]
    mass_center = get_mass_center(sample_sequence[0])
    central_location_array = np.zeros([512, 512, 512], 'float32')
    # assign central locations
    sequence_length = len(sample_sequence)
    for index in range(sequence_length):
        item = sample_sequence[index]
        central_location_array[item['center_location']] = index + 1
    # assign mass center
    if central_location_array[mass_center] > 0:
        central_location_array[mass_center] = central_location_array[mass_center] + sequence_length + 100
    else:
        central_location_array[mass_center] = sequence_length + 100

    new_central_location_array = random_flip_rotate_swap(central_location_array, False, labels, reverse=reverse)
    for item in sample_sequence:
        operate_on_arrays_in_item(item, labels, reverse)

    new_sample_sequence = []
    new_mass_center = None
    new_central_loc_list = Functions.get_location_list(np.where(new_central_location_array > 0))
    for central_loc in new_central_loc_list:
        index_plus_one = round(new_central_location_array[central_loc])
        if index_plus_one > sequence_length + 10:  # the central_loc is the overall mass center
            new_mass_center = central_loc
    assert new_mass_center is not None
    for central_loc in new_central_loc_list:
        index_plus_one = round(new_central_location_array[central_loc])
        if index_plus_one > sequence_length + 100:
            index_plus_one = index_plus_one - sequence_length - 100
        if index_plus_one > sequence_length:  # mass center
            continue
        new_sample_sequence.append(update_central_loc_and_loc_offset(
            sample_sequence[index_plus_one - 1], new_mass_center, central_loc))

    new_sample["sample_sequence"] = new_sample_sequence

    return new_sample


def get_mass_center(item):
    central_location_offset = item['location_offset']
    central_location = item['center_location']

    mass_center = (central_location[0] - central_location_offset[0], central_location[1] - central_location_offset[1],
                   central_location[2] - central_location_offset[2])

    return mass_center


def operate_on_arrays_in_item(item, labels, reverse):
    key_list = list(item.keys())
    for key in key_list:
        if item[key] is None or type(item[key]) in [list, float, tuple]:
            continue
        item[key] = random_flip_rotate_swap(item[key], False, labels, reverse=reverse)


def update_central_loc_and_loc_offset(item, new_mass_center, new_central_loc):
    new_offset = (new_central_loc[0] - new_mass_center[0], new_central_loc[1] - new_mass_center[1],
                  new_central_loc[2] - new_mass_center[2])
    item['location_offset'] = new_offset
    item['center_location'] = new_central_loc
    return item


if __name__ == '__main__':
    import pulmonary_embolism_v3.utlis.sequence_rescaled_ct_converter as converter
    import visualization.visualize_3d.visualize_stl as stl

    semantic_key = 'blood_region'

    test_sample = Functions.pickle_load_object('/data_disk/pulmonary_embolism/training_dataset_new/'
                                               'combine_ready_not_denoise/disk41-2_2020-05-05.pickle')

    ct_array = converter.reconstruct_rescaled_ct_from_sample_sequence(test_sample["sample_sequence"], (4, 4, 5),
                                                                      key=semantic_key)
    # Functions.image_show(ct_array[:, :, 256])
    # stl.visualize_numpy_as_stl(ct_array)

    augment_label = get_labels()
    print(augment_label)
    swap_label = augment_label[2]

    sample_new = random_flip_rotate_swap_sample(test_sample, labels=augment_label, reverse=False)

    ct_array_new = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_new["sample_sequence"], (4, 4, 5),
                                                                          key=semantic_key)
    # Functions.image_show(ct_array_new[:, :, 256])
    stl.visualize_numpy_as_stl(ct_array_new)
    exit()

    sample_reversed = random_flip_rotate_swap_sample(sample_new, labels=augment_label, reverse=True)

    ct_reverse = converter.reconstruct_rescaled_ct_from_sample_sequence(sample_reversed["sample_sequence"], (4, 4, 5),
                                                                        key=semantic_key)

    Functions.image_show(ct_reverse[:, :, 256])
    stl.visualize_numpy_as_stl(ct_reverse)

    differ = ct_array - ct_reverse
    Functions.image_show(differ[:, :, 256])

    print(np.sum(np.abs(ct_array - ct_reverse)))
