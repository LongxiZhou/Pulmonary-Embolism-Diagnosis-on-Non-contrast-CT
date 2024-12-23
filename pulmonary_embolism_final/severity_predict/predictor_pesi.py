import Tool_Functions.load_in_data as load_in_data
import Tool_Functions.Functions as Functions
import numpy as np
import os


def reformat_pesi_data(path_csv='/data_disk/pulmonary_embolism_final/Patient-ID_PESI.csv'):
    data_array = load_in_data.convert_csv_to_numpy(path_csv)
    patient_id_pesi_dict = {}
    for line in data_array:
        patient_id_pesi_dict[line[0]] = line[-1]

    return patient_id_pesi_dict


def classify_severity(pesi_score):
    if pesi_score <= 65:
        return 1
    if pesi_score <= 85:
        return 2
    if pesi_score <= 105:
        return 3
    if pesi_score <= 125:
        return 4
    return 5


def form_training_data(path_csv='/data_disk/pulmonary_embolism_final/Patient-ID_PESI.csv',
                       dir_cta_statistics='/data_disk/pulmonary_embolism_final/pickle_objects/statistic_clot_cta'):
    patient_id_pesi_dict = reformat_pesi_data(path_csv)
    feature_name_list = ['pesi_score', 'pesi_severity', 'avr', 'acr', 'acv', 'vcr', 'vcv']

    data_array = []

    for patient in patient_id_pesi_dict.keys():
        path_cta_stat = os.path.join(dir_cta_statistics, patient + '.pickle')
        cta_statistic = Functions.pickle_load_object(path_cta_stat)
        data_patient = [patient_id_pesi_dict[patient], classify_severity(patient_id_pesi_dict[patient]),
                        cta_statistic['avr'], cta_statistic['acr'], cta_statistic['acv'], cta_statistic['vcr'],
                        cta_statistic['vcv']]
        data_array.append(data_patient)
        print(data_patient)

    return np.array(data_array, 'float32')


if __name__ == '__main__':
    form_training_data()
