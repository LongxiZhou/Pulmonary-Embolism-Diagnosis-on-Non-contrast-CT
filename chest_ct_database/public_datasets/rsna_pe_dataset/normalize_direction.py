from ct_direction_check.chest_ct.inference import *


def normalize_direction_dataset(fold=(0, 3)):
    # modify to fit your rescaled ct dataset
    Functions.set_visible_device('1')
    top_dict = '/data_disk/RSNA-PE_dataset/rescaled_ct'

    fn_list = os.listdir(top_dict)[fold[0]:: fold[1]]

    model_ = load_model()
    processed = 0
    for fn in fn_list:
        print("processing", fn, processed, '/', len(fn_list))
        array = np.load(os.path.join(top_dict, fn))['array']
        new_array, class_id = cast_to_standard_direction(
            array, model=model_, show_image=False, show_prob=True,
            deep_copy=False, return_original_direction_class=True)
        if not class_id == 0:
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
    # step 1, cast into direction id 0
    normalize_direction_dataset()

    # step 2, predict again the direction for new data, for some wrong scan, the direction is not 0, should be remove!
    form_direction_probability_dataset()
