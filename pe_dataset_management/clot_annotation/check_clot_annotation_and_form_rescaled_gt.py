"""

"""
import os
import pe_dataset_management.basic_functions as basic_functions
import format_convert.dcm_np_converter_new as convert_to_rescaled
import numpy as np
import Tool_Functions.Functions as Functions


def get_path_clot_gt(fn_folder):
    file_names = os.listdir(fn_folder)
    mha_file_name_list = []
    for file in file_names:
        if '.mha' in file:
            mha_file_name_list.append(file)
    assert len(mha_file_name_list) == 1
    return os.path.join(fn_folder, mha_file_name_list[0])


def show_clot_gt_on_rescaled(rescaled_cta, rescaled_clot_gt, show=True):
    mass_center_z = Functions.get_mass_center_for_binary(rescaled_clot_gt)[2]
    mass_center_z = int(mass_center_z)
    image_cta = np.clip(
        rescaled_cta[:, :, mass_center_z], Functions.change_to_rescaled(-800), Functions.change_to_rescaled(500))
    image = Functions.merge_image_with_mask(image_cta, rescaled_clot_gt[:, :, mass_center_z], show=show)
    return image


if __name__ == '__main__':
    top_dict_clot_annotation = '/data_disk/CTA-CT_paired-dataset/transfer/61/'

    image_save_dict_check = '/data_disk/CTA-CT_paired-dataset/transfer/61_check'

    fn_list = os.listdir(top_dict_clot_annotation)

    for fn in fn_list:
        print(fn)
        image_save_path = os.path.join(image_save_dict_check, fn + '.png')

        if os.path.exists(image_save_path):
            print("processed")
            continue

        dataset_cta_fn = basic_functions.find_patient_id_dataset_correspondence(fn, strip=True)[0]
        dcm_folder_cta = basic_functions.find_original_dcm_folders(fn)[0]
        clot_gt_path = get_path_clot_gt(os.path.join(top_dict_clot_annotation, fn))
        clot_gt_rescaled = convert_to_rescaled.establish_rescaled_mask(clot_gt_path, source_dcm_dict=dcm_folder_cta,
                                                                       cast_to_binary=False)
        clot_gt_rescaled = np.array(clot_gt_rescaled > 0, 'float32')
        rescaled_cta_denoise = np.load(os.path.join(dataset_cta_fn, 'rescaled_ct-denoise', fn + '.npz'))['array']

        clot_save_dict = os.path.join(dataset_cta_fn, "clot_gt")
        Functions.save_np_array(clot_save_dict, fn + '.npz', clot_gt_rescaled, compress=True)
        print("\nsaving clot rescaled to path:", os.path.join(clot_save_dict, fn + '.npz'))

        cta_with_clot = show_clot_gt_on_rescaled(rescaled_cta_denoise, clot_gt_rescaled, show=False)
        Functions.image_save(cta_with_clot, image_save_path, dpi=300)
