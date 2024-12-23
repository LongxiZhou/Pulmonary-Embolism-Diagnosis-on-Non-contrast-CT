import Tool_Functions.Functions as Functions
from classic_models.XG_boost.xg_model import xg_regression
import numpy as np
import os


data_set_name_list = ['non_PE_four_center_data', 'non_PE_mudanjiang', 'non_PE_rad', 'non_PE_xwzc', 'non_PE_yidayi',
                      'PE_paired_dataset', 'unknown_paired_dataset']


name_list_healthy = ['non_PE_four_center_data', 'non_PE_xwzc']
name_list_covid = ['non_PE_mudanjiang', 'non_PE_yidayi']
name_list_average_radiology = ['non_PE_rad', ]
name_list_non = name_list_healthy + name_list_covid + name_list_average_radiology
name_list_non_chinese = name_list_healthy + name_list_covid
name_list_pe = ['PE_paired_dataset', ]

top_dir_statistic = '/data_disk/pulmonary_embolism_final/statistic_cta_confirm'


def form_data_array(list_non, list_pe, model_type='new_model', augment=True, trim_length=4000, func_name_label=None):
    dir_statistic = os.path.join(top_dir_statistic, model_type)
    if augment:
        dir_statistic = os.path.join(dir_statistic, 'augment_' + 'trim_' + str(trim_length))
    else:
        dir_statistic = os.path.join(dir_statistic, 'not_augment_' + 'trim_' + str(trim_length))

    feature_name_list = ['gt', 'avr', 'acr', 'acv', 'vcr', 'vcv']
    data_array = []

    def process_dataset(pe_label, dataset_name):
        sub_dir_dataset = os.path.join(dir_statistic, dataset_name)
        for name_pickle in os.listdir(sub_dir_dataset):
            line = []
            if func_name_label is None:
                line.append(pe_label)
            else:
                line.append(name_pickle[:-7])
            statistic = Functions.pickle_load_object(os.path.join(sub_dir_dataset, name_pickle))
            line.append(statistic['avr'])
            line.append(statistic['acr'])
            line.append(statistic['acv'])
            line.append(-statistic['vcr'])
            line.append(-statistic['vcv'])

            if not 0.1 < statistic['avr'] < np.inf:
                continue

            data_array.append(line)

    for name in list_non:
        process_dataset(0, name)
    for name in list_pe:
        process_dataset(1, name)

    return np.array(data_array, 'float32'), feature_name_list


if __name__ == '__main__':
    array_, features_ = form_data_array(name_list_covid, name_list_pe, model_type='new_model', augment=False)
    print(np.shape(array_))

    model, predicted, gt = xg_regression(array_, features_, 0, [1, 2, 3, 4, 5], test_num=50)

    from sklearn import metrics
    from matplotlib import pyplot as plt
    from matplotlib import pyplot as plt

    auc = metrics.roc_auc_score(array_[:, 0], predicted)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(array_[:, 0], predicted)

    index = 0
    while not true_positive_rate[index] >= 0.5:
        index += 1
    print("FPR at recall 0.5", false_positive_rate[index])

    index = 0
    while not false_positive_rate[index] >= 0.03:
        index += 1
    print("TPR at precision 0.97", true_positive_rate[index])

    plt.figure(figsize=(12, 12), dpi=300)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve for Diagnosis PE from Non-contrast CT (Simulate & Real Clots)")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.show()
