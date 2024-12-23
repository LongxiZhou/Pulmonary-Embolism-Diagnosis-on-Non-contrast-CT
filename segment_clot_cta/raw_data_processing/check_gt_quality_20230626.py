import numpy as np
import Tool_Functions.Functions as Functions
import os


def visualize_one_scan(scan_name):
    print("processing:", scan_name, processed_count, '/', len(fn_list))
    if len(scan_name) <= 4:
        scan_name = scan_name + '.npz'
    if len(scan_name) > 4:
        if not scan_name[-4:] == '.npz':
            scan_name = scan_name + '.npz'

    path_x = os.path.join(visualization_save_dict, Functions.strip_suffix(scan_name) + '_x.png')
    path_y = os.path.join(visualization_save_dict, Functions.strip_suffix(scan_name) + '_y.png')
    path_z = os.path.join(visualization_save_dict, Functions.strip_suffix(scan_name) + '_z.png')

    if os.path.exists(path_z):
        print("processed")
        return None

    clot_gt = np.load(os.path.join(clot_gt_dict, scan_name))['array']
    rescaled_cta = np.load(os.path.join(rescaled_cta_dict, scan_name))['array']

    mass_center = Functions.get_mass_center_for_binary(clot_gt)

    rescaled_cta = np.clip(rescaled_cta + 0.5, 0.2, 1.2)

    x, y, z = np.array(mass_center, 'int32')

    print(x, y, z)

    Functions.merge_image_with_mask(rescaled_cta[x, :, :], clot_gt[x, :, :], show=False, save_path=path_x, dpi=300)
    Functions.merge_image_with_mask(rescaled_cta[:, y, :], clot_gt[:, y, :], show=False, save_path=path_y, dpi=300)
    Functions.merge_image_with_mask(rescaled_cta[:, :, z], clot_gt[:, :, z], show=False, save_path=path_z, dpi=300)


clot_gt_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_gt_waiting_check'
rescaled_cta_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_ct-denoise'
visualization_save_dict = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/' \
                          'PE_CTA_no_gt/visualization/check_rescaled_gt'

fn_list = os.listdir(clot_gt_dict)

processed_count = 0
for fn in fn_list:
    visualize_one_scan(fn)
    processed_count += 1


"""
low_quality scan [11.17p11, patient-id-11.17p11, patient-id-23752144, patient-id-p94, 
Z104-2022-06-10, Z110-2022-05-02. Z118]
"""
