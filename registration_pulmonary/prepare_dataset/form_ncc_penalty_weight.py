"""
the penalty weight is formed by three parts

1) airway plus blood vessel, 2) pulmonary plus heart regions, 3) any region, 4) branch level for blood

the penalty weight sum for
1) is 0.15 Unit; for 2) is 0.25 Unit; for 3) is 1 Unit
for 4) is around 0.1 Unit (apply blood branch map to focus on small blood vessels)

total penalty is around 1.5 Unit

each voxel can belongs to different parts (will get weights from different source)
"""
import numpy as np


def calculate_penalty_weight(blood_vessel_mask, airway_mask, lung_mask, heart_mask, branch_array, base_penalty=0.1):
    """

    :param blood_vessel_mask:
    :param airway_mask:
    :param lung_mask:
    :param heart_mask:
    :param branch_array:
    :param base_penalty:
    :return: penalty weight in numpy float32, same shape with input masks
    """
    shape = np.shape(blood_vessel_mask)

    penalty_base = np.product(shape) * base_penalty
    penalty_weight = np.zeros(np.shape(blood_vessel_mask), 'float32') + base_penalty

    lung_heart_mask = np.array(lung_mask + heart_mask > 0, 'float32')
    penalty_weight = penalty_weight + lung_heart_mask * (penalty_base * 0.25 / np.sum(lung_heart_mask))

    blood_airway_mask = np.array(blood_vessel_mask + airway_mask > 0, 'float32')
    penalty_weight = penalty_weight + blood_airway_mask * (penalty_base * 0.15 / np.sum(blood_airway_mask))

    penalty_weight = penalty_weight + np.clip(branch_array, 0, 10)

    return penalty_weight


def calculate_penalty_weight_pe_paired_dataset(scan_name, top_dict_pe_dataset='/data_disk/CTA-CT_paired-dataset',
                                               show=False):
    """

    use non-contrast to form penalty.

    :param scan_name:
    :param top_dict_pe_dataset:
    :param show
    :return:
    """
    from pe_dataset_management.basic_functions import find_patient_id_dataset_correspondence
    import os

    if len(scan_name) < 4:
        scan_name = scan_name + '.npz'
    if not scan_name[-4:] == '.npz':
        scan_name = scan_name + '.npz'

    _, top_dict_non_contrast = find_patient_id_dataset_correspondence(
        scan_name, strip=True, top_dict=top_dict_pe_dataset)

    top_dict_semantics = os.path.join(top_dict_non_contrast, 'semantics')
    top_dict_branch = os.path.join(top_dict_non_contrast, 'depth_and_center-line')

    vessel = np.load(os.path.join(top_dict_semantics, 'blood_mask_high_recall/', scan_name))['array']
    branch_mask = np.load(os.path.join(top_dict_branch, 'high_recall_blood_branch_map/', scan_name))['array']
    airway = np.load(os.path.join(top_dict_semantics, 'airway_mask/', scan_name))['array']
    lung = np.load(os.path.join(top_dict_semantics, 'lung_mask/', scan_name))['array']
    heart = np.load(os.path.join(top_dict_semantics, 'heart_mask/', scan_name))['array']

    penalty = calculate_penalty_weight(vessel, airway, lung, heart, branch_mask)

    if show:
        print('average penalty weight', np.mean(penalty))  # should around 0.15
        import Tool_Functions.Functions as Functions
        for i in range(200, 400, 10):
            Functions.image_show(penalty[:, :, i])

    return penalty


if __name__ == '__main__':

    calculate_penalty_weight_pe_paired_dataset('Z101.npz')
    exit()
