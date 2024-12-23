import Tool_Functions.load_in_data as load_in_data
import Tool_Functions.Functions as Functions
import numpy as np
import os


def form_mrn_info_dict(path_csv='/data_disk/pulmonary_embolism_final/mrn-info.csv'):
    data_array = load_in_data.convert_csv_to_numpy(path_csv)

    mrn_value_dict = {}

    for line in data_array:
        mrn_value_dict[line[1]] = {'SBP': float(line[13]), 'TNI': float(line[18]), 'RVD': float(line[19]),
                                   'D-Dimer': float(line[20])}

    return mrn_value_dict


def reformat_pesi_data(path_csv='/data_disk/pulmonary_embolism_final/Patient-ID_PESI.csv'):
    data_array = load_in_data.convert_csv_to_numpy(path_csv)
    patient_id_mrn_pesi_dict = {}
    for line in data_array:
        patient_id_mrn_pesi_dict[line[0]] = {'MRN': line[1], 'PESI': line[-1]}

    return patient_id_mrn_pesi_dict


def form_final_data_dict():
    """

    :return: {
    'patient-id-XXX':
    {'MRN':,
    'cta_metrics': {'avr':, 'acr':, 'acv':, 'vcr', 'vcv':},
    'severity': {'PESI':, 'SBP':, 'TNI':, 'RVD':, 'D-Dimer': }
    },
    ...}
    """
    mrn_value_dict = form_mrn_info_dict()
    patient_id_mrn_pesi_dict = reformat_pesi_data()
    dir_statistic_cta = '/data_disk/pulmonary_embolism_final/pickle_objects/statistic_clot_cta'

    final_dict = {}

    for patient_id in patient_id_mrn_pesi_dict.keys():
        mrn = patient_id_mrn_pesi_dict[patient_id]['MRN']

        sub_sub_dict_severity = {
            'PESI': patient_id_mrn_pesi_dict[patient_id]['PESI'],
            'SBP': mrn_value_dict[mrn]['SBP'],
            'TNI': mrn_value_dict[mrn]['TNI'],
            'RVD': mrn_value_dict[mrn]['RVD'],
            'D-Dimer':  mrn_value_dict[mrn]['D-Dimer'],
        }

        sub_dict_id = {
            'MRN': mrn,
            'cta_metrics': Functions.pickle_load_object(os.path.join(dir_statistic_cta, patient_id + '.pickle')),
            'severity': sub_sub_dict_severity
        }

        final_dict[patient_id] = sub_dict_id

    return final_dict


if __name__ == '__main__':
    final_ = form_final_data_dict()
    for p_id, value in final_.items():
        if not value['severity']['SBP'] > 90:
            print(p_id, value['severity'])

            path_a = os.path.join('/data_disk/pulmonary_embolism_final/statistic_cta_confirm/new_model/not_augment_trim_4000/PE_paired_dataset', p_id + '.pickle')
            path_b = os.path.join('/data_disk/pulmonary_embolism_final/statistic_cta_confirm/new_model/not_augment_trim_4000/unknown_paired_dataset', p_id + '.pickle')

            if os.path.exists(path_a):
                print(Functions.pickle_load_object(path_a))
            if os.path.exists(path_b):
                print(Functions.pickle_load_object(path_b))

    print(len(final_))
