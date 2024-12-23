import Tool_Functions.Functions as Functions
import Tool_Functions.file_operations as file_operations
import os


fn_list = os.listdir('/data_disk/RSNA-PE_dataset/pickle_objects/direction_probability')
processed = 0
for fn in fn_list:
    # print(processed, '/', len(fn_list))
    good = True
    direction_probability_0 = Functions.pickle_load_object(
        '/data_disk/RSNA-PE_dataset/pickle_objects/direction_probability/' + fn)[0]
    if not direction_probability_0['class_id'] == 0:
        good = False  # highly likely this is not chest scan, missing lots of slices, wrong resolution, etc.
    if not direction_probability_0['predicted_probability'] > 0.99:
        good = False  # severe lesions, wrong resolution, missing slice, etc.

    if not good:
        rescaled_ct_path = os.path.join('/data_disk/RSNA-PE_dataset/rescaled_ct', fn[:-7] + '.npz')
        if os.path.exists(rescaled_ct_path):
            print(fn, direction_probability_0)
            file_operations.remove_path_or_directory(rescaled_ct_path, show=True)

    processed += 1
