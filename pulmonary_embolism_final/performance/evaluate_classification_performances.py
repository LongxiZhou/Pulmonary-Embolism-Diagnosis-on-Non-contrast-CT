import Tool_Functions.file_operations as file_operations
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt.stratify_gt_quality import \
    get_quality_of_scan_name
import os


def form_final_dataframe():
    """
    each case, have these metrics:

    pe_condition: string in ['PE_pair_quality_0', 'PE_pair_quality_1',
                            'PE_pair_quality_2', 'PE_pair_quality_3', 'Long_Time', 'Not_PE']
    scan_name: string like 'Z111'
    a-v_clot_ratio_v0: float
    a-v_clot_ratio_v0_strict: float
    artery_clot_ratio_v0: float
    artery_clot_ratio_v0_strict: float
    artery_clot_volume_v0: float
    artery_clot_volume_v0_strict: float

    a-v_clot_ratio_v1: float
    a-v_clot_ratio_v1_strict: float
    artery_clot_ratio_v1: float
    artery_clot_ratio_v1_strict: float
    artery_clot_volume_v1: float
    artery_clot_volume_v1_strict: float

    a-v_clot_ratio_v2: float
    a-v_clot_ratio_v2_strict: float
    artery_clot_ratio_v2: float
    artery_clot_ratio_v2_strict: float
    artery_clot_volume_v2: float
    artery_clot_volume_v2_strict: float


    :return:
    """
    pass


def performance_on_paired_dataset(test_id=None, policy=(0, 1, 2, 3), metric_key='v0', apply_strict_mask=False,
                                  top_dict_metric='/data_disk/pulmonary_embolism_final/statistics/'
                                                  'simulation_only_not_augment_trim_3000',
                                  min_artery_ratio=0, min_clot_volume=0):
    """

    :param min_clot_volume:
    :param min_artery_ratio:
    :param test_id:
    :param policy:
    0, good_pair_good_registration, 1, good_pair_excellent_registration,
    2, excellent_pair_good_registration, 3, excellent_pair_excellent_registration

    :param metric_key:
    v0, the guide mask is blood vessel mask
    v1, the guide mask is blood vessel mask with depth >= 3
    v2, the guide mask is the predicted blood region mask
    :param apply_strict_mask:
    if True, the predicted clot mask will be trimmed by the predicted blood region mask

    :param top_dict_metric:

    :return:
    """
    if type(policy) is int:
        policy = [policy, ]
    if policy is None:
        policy = [0, 1, 2, 3]
    if type(test_id) is int:
        test_id = [test_id]
    if test_id is None:
        test_id = [0, 1, 2, 3, 4]

    if 'clot_av_paired_dataset' not in top_dict_metric:
        top_dict_metric = os.path.join(top_dict_metric, 'clot_av_paired_dataset')

    if apply_strict_mask:
        metric_key = metric_key + '_strict'

    fn_good_pair_good_registration, fn_good_pair_excellent_registration, \
    fn_excellent_pair_good_registration, fn_excellent_pair_excellent_registration = get_quality_of_scan_name()

    policy_fn_dict = {0: fn_good_pair_good_registration, 1: fn_good_pair_excellent_registration,
                      2: fn_excellent_pair_good_registration, 3: fn_excellent_pair_excellent_registration}

    fn_list = []
    for policy_key in policy:
        fn_list = fn_list + list(policy_fn_dict[policy_key])

    name_list_final = []
    for fn in fn_list:
        name = fn + '.pickle'
        ord_sum = 0
        for char in name:
            ord_sum += ord(char)
        if ord_sum % 5 in test_id:
            name_list_final.append(name)

    value_list = []
    for name in name_list_final:
        path_static = os.path.join(top_dict_metric, name)
        values = read_in_statistic(path_static, metric_key)
        if min_artery_ratio > 0:
            if values[1] < min_artery_ratio:
                continue
        if min_clot_volume > 0:
            if values[2] < min_clot_volume:
                continue
        value_list.append(values[0])

    print("paired dataset output:", len(value_list), "cases")

    return value_list


def read_in_statistic(path_static, key):
    pickle_object = Functions.pickle_load_object(path_static)
    return pickle_object[key]  # relative a-v clot ratio, artery clot volume ratio, artery clot volume in mm^3


def performance_on_non_pe_dataset(metric_key='v0', apply_strict_mask=False,
                                  top_dict_metric='/data_disk/pulmonary_embolism_final/statistics/'
                                                  'simulation_only_not_augment_trim_3000',
                                  min_artery_ratio=0.001, min_clot_volume=50, num_return=None,
                                  dataset_non_pe=('clot_av_rad', 'clot_av_chinese')):
    if dataset_non_pe == ('clot_av_healthy', ):
        dataset_non_pe = ('clot_av_healthy_four_center', 'clot_av_healthy_xwzc')

    if type(dataset_non_pe) is str:
        if dataset_non_pe not in top_dict_metric:
            top_dict_metric = os.path.join(top_dict_metric, dataset_non_pe)
    else:
        value_list = []
        for sub_dataset in dataset_non_pe:
            value_list = value_list + performance_on_non_pe_dataset(
                metric_key, apply_strict_mask, top_dict_metric,
                min_artery_ratio, min_clot_volume, num_return, sub_dataset)
        return value_list

    if apply_strict_mask:
        metric_key = metric_key + '_strict'

    name_list = os.listdir(top_dict_metric)
    name_list.sort()
    if num_return is not None:
        name_list = name_list[0: num_return]

    value_list = []
    for name in name_list:
        path_static = os.path.join(top_dict_metric, name)
        values = read_in_statistic(path_static, metric_key)
        if min_artery_ratio > 0:
            if values[1] < min_artery_ratio:
                continue
        if min_clot_volume > 0:
            if len(values) > 2:
                if values[2] < min_clot_volume:
                    continue
        value_list.append(values[0])

    value_list.sort()
    value_list.reverse()

    return value_list


def split_chinese_metric(top_dict_metric):
    """
    split ./clot_av_chinese into
    ./clot_av_healthy_xwzc  ./clot_av_healthy_four_center  ./clot_av_COVID_mudanjinag  ./clot_av_COVID_yidayi

    :param top_dict_metric:
    :return:
    """
    assert 'clot_av_chinese' in top_dict_metric
    fn_list = os.listdir(top_dict_metric)  # end with '.pickle'

    fn_list_mudanjiang = os.listdir(
        '/data_disk/rescaled_ct_and_semantics/rescaled_ct/COVID-19/mudanjiang')  # end with '.npz'
    fn_list_yidayi = os.listdir('/data_disk/rescaled_ct_and_semantics/rescaled_ct/COVID-19/yidayi')
    fn_list_four_center = os.listdir('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/four_center_data')
    fn_list_xwzc = os.listdir('/data_disk/rescaled_ct_and_semantics/rescaled_ct/healthy_people/xwzc')

    father_dict = Functions.get_father_dict(top_dict_metric)

    save_dict_mudanjiang = os.path.join(father_dict, 'clot_av_COVID_mudanjiang')
    save_dict_yidayi = os.path.join(father_dict, 'clot_av_COVID_yidayi')
    save_dict_xwzc = os.path.join(father_dict, 'clot_av_healthy_xwzc')
    save_dict_four_center = os.path.join(father_dict, 'clot_av_healthy_four_center')

    for fn in fn_list:
        source_path = os.path.join(top_dict_metric, fn)
        if fn[:-7] + '.npz' in fn_list_mudanjiang:
            save_path = os.path.join(save_dict_mudanjiang, fn)
        elif fn[:-7] + '.npz' in fn_list_yidayi:
            save_path = os.path.join(save_dict_yidayi, fn)
        elif fn[:-7] + '.npz' in fn_list_xwzc:
            save_path = os.path.join(save_dict_xwzc, fn)
        else:
            assert fn[:-7] + '.npz' in fn_list_four_center
            save_path = os.path.join(save_dict_four_center, fn)
        file_operations.copy_file_or_dir(source_path, save_path)


if __name__ == '__main__':
    from visualization.visualize_distribution.distribution_analysis import distribution_plot

    top_dict_statistics = '/data_disk/pulmonary_embolism_final/statistics/with_gt_not_augment_trim_4000'

    auc = True
    show = True
    save_path = '/data_disk/pulmonary_embolism_final/pictures/temp_2.svg'

    value_dict = {
        # 'v1 pe simulate': performance_on_paired_dataset(test_id=0, metric_key='v1', top_dict_metric='/data_disk/pulmonary_embolism_final/statistics/no_augmentation/simulation_only'),
        'PE Non-contrast': performance_on_paired_dataset(test_id=None, metric_key='v0', policy=None,
                                                         top_dict_metric=top_dict_statistics,
                                                         apply_strict_mask=False),
        # 'v1 non pe simulate': performance_on_rad_dataset(metric_key='v1', top_dict_metric='/data_disk/pulmonary_embolism_final/statistics/no_augmentation/simulation_only', num_return=500),
        'Non-PE Non-contrast': performance_on_non_pe_dataset(metric_key='v0', top_dict_metric=top_dict_statistics,
                                                             num_return=None,
                                                             apply_strict_mask=False,
                                                             dataset_non_pe=('clot_av_rad', 'clot_av_chinese'))
    }

    if auc:

        y_true = []
        y_score = []
        for value in value_dict['PE Non-contrast']:
            if not value > 0 or value > 1000:
                y_true.append(1)
                y_score.append(1000)
                continue
            y_true.append(1)
            y_score.append(value)
        for value in value_dict['Non-PE Non-contrast']:
            if not value > 0 or value > 1000:
                continue
            y_true.append(0)
            y_score.append(value)

        from sklearn import metrics
        from matplotlib import pyplot as plt

        auc = metrics.roc_auc_score(y_true, y_score)

        false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_true, y_score)

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
        if not show:
            plt.savefig(save_path)
        else:
            plt.show()

    else:

        distribution_plot(value_dict, show_data_points=True, showfliers=False, y_range_show=(0, 100),
                          method='box_plot', y_label='Artery-Vein Clot Probability Ratio',
                          title='Detect Pulmonary Embolism from Non-contrast CT (Simulate & Real Clots)',
                          save_path=save_path)
