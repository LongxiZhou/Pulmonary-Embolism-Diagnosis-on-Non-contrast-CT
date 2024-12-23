import xgboost as xgb
import xgbfir
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import numpy as np
np.set_printoptions(suppress=True)


def delete_rows(original_array, row_list):
    shape = np.shape(original_array)
    return_array = np.zeros((shape[0] - len(row_list), shape[1]), 'float32')
    pointer = 0
    for i in range(shape[0]):
        if i in row_list:
            continue
        return_array[pointer, :] = original_array[i, :]
        pointer += 1
    return return_array


def xg_regression(data_array, feature_name_list, target_column, data_column_list,
                  test_num=1, evaluate=True, save_report_path=None, show=True):
    """
    :param data_array: array in shape [num_samples, values]:
    :param feature_name_list: the name for each value
    :param test_num: how many samples are used as test set
    :param save_report_path: if None, do not save performance report.
    :param target_column: a int, which column of the data array are considered as gt?
    :param data_column_list: a tuple of int, which column are used as data?
    :param evaluate: report the performance of the model_guided
    :param show: whether print out the intermediate results
    :return: the model_guided
    """
    print(len(np.shape(data_array)), len(feature_name_list))
    assert len(np.shape(data_array)) == 2 and np.shape(data_array)[1] == len(feature_name_list)

    feature_names = []
    for column in data_column_list:
        feature_names.append(feature_name_list[column])
    rows = np.shape(data_array)[0]
    columns = len(data_column_list)
    data = np.zeros((rows, columns), 'float32')
    pointer = 0
    for column in data_column_list:
        data[:, pointer] = data_array[:, column]
        pointer += 1
    data_set = {'target': data_array[:, target_column], 'feature_names': feature_names,
                'data': data}
    if show:
        print("there are:", rows, "samples and:", columns, "features")
        print("the target feature being predict is:", feature_name_list[target_column])
        print("the ground truth is:", data_array[:, target_column])
        print("the features are:", feature_names)
        print("the data for the first row:")
        print(data[0, :])

    model = xgb.XGBRegressor().fit(data_set['data'], data_set['target'])
    if save_report_path is not None:
        xgbfir.saveXgbFI(model, feature_names=data_set['feature_names'], OutputXlsxFile=save_report_path)

    if show:
        print('feature importance:', model.feature_importances_)

    if not evaluate:
        return model

    predicted = []
    gt = data_array[:, target_column]

    temp_data = np.zeros((rows - test_num, columns), 'float32')
    temp_target = np.zeros((rows - test_num,), 'float32')

    for test_start in Functions.iteration_with_time_bar(range(0, rows, test_num)):

        if test_start + test_num < rows:
            temp_data[0: test_start, :] = data[0: test_start, :]
            temp_target[0: test_start] = gt[0: test_start]
            test_num_this_fold = test_num
            temp_data[test_start::, :] = data[test_start + test_num::, :]
            temp_target[test_start::] = gt[test_start + test_num::]
        else:
            temp_data[:, :] = data[0: rows - test_num, :]
            temp_target = gt[0: rows - test_num]
            test_num_this_fold = rows - test_start

        temp_model = xgb.XGBRegressor().fit(temp_data, temp_target)
        predicted = predicted + list(temp_model.predict(data[test_start: test_start + test_num_this_fold, :]))

    """
    from Tool_Functions.fitting_and_simulation import up_sample_linearly_paired_samples
    gt, predicted = up_sample_linearly_paired_samples(gt, predicted, new_sample_number=120, y_error=0.2,
                                                      x_bound=(0.2, 0.95), selected_x_interval=0.1)
    """

    return model, np.array(predicted), np.array(gt)


def xg_regression_old(data_array, feature_name_list, target_column, data_column_list,
                      test_num=1, evaluate=True, save_report_path=None, show=True):
    """
    :param data_array: array in shape [num_samples, values]:
    :param feature_name_list: the name for each value
    :param test_num: how many samples are used as test set
    :param save_report_path: if None, do not save performance report.
    :param target_column: a int, which column of the data array are considered as gt?
    :param data_column_list: a tuple of int, which column are used as data?
    :param evaluate: report the performance of the model_guided
    :param show: whether print out the intermediate results
    :return: the model_guided
    """
    print(len(np.shape(data_array)), len(feature_name_list))
    assert len(np.shape(data_array)) == 2 and np.shape(data_array)[1] == len(feature_name_list)

    feature_names = []
    for column in data_column_list:
        feature_names.append(feature_name_list[column])
    rows = np.shape(data_array)[0]
    columns = len(data_column_list)
    data = np.zeros((rows, columns), 'float32')
    pointer = 0
    for column in data_column_list:
        data[:, pointer] = data_array[:, column]
        pointer += 1
    data_set = {'target': data_array[:, target_column], 'feature_names': feature_names,
                'data': data}
    if show:
        print("there are:", rows, "samples and:", columns, "features")
        print("the target feature being predict is:", feature_name_list[target_column])
        print("the ground truth is:", data_array[:, target_column])
        print("the features are:", feature_names)
        print("the data for the first row:")
        print(data[0, :])

    model = xgb.XGBRegressor().fit(data_set['data'], data_set['target'])
    if not evaluate:
        if save_report_path is not None:
            xgbfir.saveXgbFI(model, feature_names=data_set['feature_names'], OutputXlsxFile=save_report_path)
        print(model.feature_importances_)
        return model

    predicted = []
    gt = data_array[:, target_column]

    temp_data = np.zeros((rows - test_num, columns), 'float32')
    temp_target = np.zeros((rows - test_num,), 'float32')

    for test_start in Functions.iteration_with_time_bar(range(0, rows, test_num)):

        if test_start + test_num < rows:
            temp_data[0: test_start, :] = data[0: test_start, :]
            temp_target[0: test_start] = gt[0: test_start]
            test_num_this_fold = test_num
            temp_data[test_start::, :] = data[test_start + test_num::, :]
            temp_target[test_start::] = gt[test_start + test_num::]
        else:
            temp_data[:, :] = data[0: rows - test_num, :]
            temp_target = gt[0: rows - test_num]
            test_num_this_fold = rows - test_start

        temp_model = xgb.XGBRegressor().fit(temp_data, temp_target)
        predicted = predicted + list(temp_model.predict(data[test_start: test_start + test_num_this_fold, :]))

    """
    from Tool_Functions.fitting_and_simulation import up_sample_linearly_paired_samples
    gt, predicted = up_sample_linearly_paired_samples(gt, predicted, new_sample_number=120, y_error=0.2,
                                                      x_bound=(0.2, 0.95), selected_x_interval=0.1)
    """

    r_score, p = metrics.pearson_correlation_coefficient(predicted, gt)
    r_spearman, p_spearman = metrics.spearman_ranking_correlation_coefficient(predicted, gt)
    root_mean_error = metrics.norm_mean_error(predicted, gt, 2)
    abs_mean_error = metrics.norm_mean_error(predicted, gt, 1)

    base = np.array(gt, 'float32')
    base[:] = np.average(gt)
    root_mean_error_base = metrics.norm_mean_error(base, gt, 2)
    abs_mean_error_base = metrics.norm_mean_error(base, gt, 1)

    if show:
        print("pearson score:", r_score, "p:", p)
        print("spearman score:", r_spearman, "p_spearman:", p_spearman)
        print("root mean error:", root_mean_error)
        print("abs mean error:", abs_mean_error)

        print("root mean error base:", root_mean_error_base)
        print("abs mean error base:", abs_mean_error_base)

        Functions.show_data_points(
            gt, predicted, x_name='Actual Dice', y_name='Estimated Dice',
            title='Audit the Dice', data_label=None,
            save_path='/home/zhoul0a/Desktop/Audit_Dice.svg')

    difference_abs = np.abs(np.array(gt) - np.array(predicted))
    return model, r_score, p, root_mean_error, abs_mean_error, difference_abs, np.array(predicted), np.array(gt)


if __name__ == '__main__':
    array = np.load('/data_disk/Breast_Cancer/analysis/healthy_mean_std/positive_negative_relations.npy')
    feature_list = ["mean_certainty_health", "std_certainty_health", "balance_certainty"]
    data_path = None
    total_lines = np.shape(array)[0]
    delete_list_already = [68, 106, 110, 126] + [44, 86, 122] + [6, 50, 102] + [17, 22, 40, 42, 81]
    print(np.shape(array)[0] - len(delete_list_already))
    data_array_ = delete_rows(array, delete_list_already)
    xg_regression_old(data_array_, feature_list, 2, [0, 1], show=True)
    exit()
