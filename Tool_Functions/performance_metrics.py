"""
This file is for the performance metrics of all_file kinds
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import norm
import math


def get_significant(std_num):
    return (1 - norm.cdf(std_num)) * 2


def linear_fit(x, y, show=True, std_for_r=False):
    n = len(x)
    assert n == len(y)
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, n):
        sx += x[i]
        sy += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/n - sxy)/(sx*sx/n - sxx)
    b = (sy - a*sx)/n
    r = (sy*sx/n-sxy)/math.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    if show:
        print("the fitting result is: y = %10.5f x + %10.5f , r = %10.5f" % (a, b, r))
    if not std_for_r:
        return a, b, r
    std = math.sqrt((1 - r*r)/(len(x) - 2))
    return a, b, r, std


def pearson_correlation_coefficient(predict, gt):
    # for output that can be considered as float
    predict = np.array(predict, 'float32').reshape((-1,))
    gt = np.array(gt, 'float32').reshape((-1,))
    num_samples = len(predict)
    assert num_samples == len(gt)
    r, p = pearsonr(predict, gt)
    return r, p


def assign_id_and_block(sequence):
    """
    block_id: same_value have same block_id; same block_id means same_value
    index: sequence[index] == value
    :param sequence: freq list like input, element in float32, like [value_1, value_2, value_3, ...]
    :return: freq list, element [value, index, block], like [[value_1, 0, block_id], [value_2, 1, block_id], ...]
    """
    from Tool_Functions.Functions import customized_sort

    def compare(a, b):
        if a[1] > b[1]:
            return 1
        return -1

    length = len(sequence)
    index_list = np.arange(length)
    sorted_value_list = list(zip(sequence, index_list))
    sorted_value_list.sort()
    final_list = []
    block_id = 0
    previous_value = sorted_value_list[0][0]
    for i in range(length):
        value = sorted_value_list[i][0]
        index = sorted_value_list[i][1]
        if not value == previous_value:  # new block
            block_id += 1
            final_list.append([value, index, block_id])
            previous_value = value
        else:
            final_list.append([value, index, block_id])
    final_list = customized_sort(final_list, compare, reverse=False)
    return final_list


def block_propagation(gt_extend, predict_extend):
    length = len(gt_extend)
    assert length == len(predict_extend)
    assert length > 0

    def compare(item_a, item_b):
        if item_a[1] > item_b[1]:
            return 1
        return -1

    predict_extend.sort()  # now, the predicted value increasing
    for i in range(length - 1):
        item = predict_extend[i]
        item_front = predict_extend[i + 1]
        block_id = item[2]
        index_id = item[1]
        value = item[0]
        index_id_front = item_front[1]
        if gt_extend[index_id][2] == gt_extend[index_id_front][2]:
            # this means the two value has the same gt and is adjacent in ranking
            item_front[2] = block_id
            item_front[0] = value
    from Tool_Functions.Functions import customized_sort
    predict_extend = customized_sort(predict_extend, compare, reverse=False)
    return predict_extend


def spearman_ranking_correlation_coefficient(predict, gt, strict=False, show=False):
    """
    e.g. input (2, 5, 3, 1, 6, 4) (3, 6, 4, 2, 7, 5), return (1.0, 0.0) which is (r, p-value)
    :param show:
    :param predict: list like, in float, like: (value_for_patient_1, value_for_patient_2, ..., value_for_patient_n)
    :param gt: list like, in float, like: (gt_for_patient_1, gt_for_patient_2, ..., gt_for_patient_n)
    must have same length with predict
    :param strict: True for standard spearman r, False, predict [1, 2, 3] gt [1, 1, 1] results in correlation 1.0
    e.g.
    gt = [1, 2, 2, 3, 3, 3, 3, 5, 6, 6, 6]
    predict = [2, 3, 3.5, 4, 4.5, 4.2, 3.9, 6, 7, 8, 9]
    strict = True, spearman score = 0.96;  strict = False, spearman score = 1.0
    :return: spearman correlation, p-value
    """
    predict = np.array(predict, 'float32').reshape((-1,))
    gt = np.array(gt, 'float32').reshape((-1,))
    num_samples = len(predict)
    assert num_samples == len(gt)
    if strict:
        r, p = spearmanr(predict, gt)
    else:
        gt_extend = assign_id_and_block(gt)
        predict_extend = assign_id_and_block(predict)
        predict_new = block_propagation(gt_extend, predict_extend)
        predict = []
        for i in predict_new:
            predict.append(i[0])
        return spearman_ranking_correlation_coefficient(predict, gt, True, show=show)
    if show:
        print("r =", r, "   p =", p)
    return r, p


def norm_mean_error(predict, gt, order=2):
    # input list like object. e.g. predict = [1.4, 5.5, 2, 54]; gt = [1.6, 5.2, 2.7, 70];
    # for output that can be considered as float
    # order = 1, mean absolute error, MSE
    # order = 2, root mean square error, RMSE
    assert order > 0
    num_samples = len(predict)
    assert num_samples == len(gt)
    error_sum = 0
    for i in range(num_samples):
        differ = abs(predict[i] - gt[i])
        error_sum += math.pow(differ, order)
    mean_error = error_sum / num_samples
    return math.pow(mean_error, 1 / order)


def relative_norm_mean_error(predict, gt, order=2, base=0):
    # input list like object. e.g. predict = [1.4, 5.5, 2, 54]; gt = [1.6, 5.2, 2.7, 70];
    # for data that can be considered as float
    # order = 1, mean relative absolute error, MSRE
    # order = 2, root mean relative square error, RMSRE
    # base is that we divide (abs((predict[i] + gt[i]) / 2) + base) for "relative"
    assert order > 0
    num_samples = len(predict)
    assert num_samples == len(gt)
    error_sum = 0
    for i in range(num_samples):
        differ = abs(predict[i] - gt[i]) / (abs((predict[i] + gt[i]) / 2) + base)
        error_sum += math.pow(differ, order)
    mean_error = error_sum / num_samples
    return math.pow(mean_error, 1 / order)


def recall(prediction, ground_truth, threshold=0.5):
    prediction = np.array(prediction > threshold, 'float32')
    ground_truth = np.array(ground_truth > threshold, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    total_positive = np.sum(ground_truth)
    return over_lap / total_positive


def dice_score_two_class(prediction, ground_truth, beta=1, check=True, simple=False):
    """
    calculate the dice score for two classes, in this case, f1 score is identical with dice score.
    for prediction and ground_truth arrays, positive is 1, negative is 0.
    :param simple: only return dice =  2 * np.sum(pre * mask) / (np.sum(pre * pre) + np.sum(mask * mask))
    :param check:
    :param beta: the recall is considered beta times more important than precision
    :param prediction: freq numpy array, in float32
    :param ground_truth:  freq numpy array, in float32
    :return: f1_score, recall, precision
    """
    assert np.shape(prediction) == np.shape(ground_truth)
    if check:
        assert np.max(prediction) <= 1.000001, np.min(prediction) >= -0.000001
        assert np.max(ground_truth) <= 1.000001, np.min(ground_truth) >= -0.000001

    if simple:
        return 2 * np.sum(prediction * ground_truth) / (
                np.sum(prediction * prediction) + np.sum(ground_truth * ground_truth))

    prediction_array = np.array(prediction > 0.5, 'float32')
    ground_truth_array = np.array(ground_truth > 0.5, 'float32')

    true_positives = np.sum(prediction_array * ground_truth_array)
    false_positives = np.sum(np.clip(prediction_array - ground_truth_array, 0, 1))
    false_negatives = np.sum(np.clip(ground_truth_array - prediction_array, 0, 1))

    if true_positives + false_positives == 0:
        return 0, 0, 0

    if true_positives + false_negatives == 0:
        return 0, 0, 0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = (1 + beta * beta) * precision * recall / ((beta * beta * precision) + recall)

    return f1_score, recall, precision


def region_discovery_dice_z_axis(predicted_binary, gt_binary, cast_to_binary=True, show=False):
    """

    Calculate the region discovery dice for 3D arrays from z-axis
    Slice by slice get the connected regions in predicted and gt from z-axis, see whether they are overlapped
    return: number of connected region overlapped / total number of connected region, ranges in [0, 1]

    :param predicted_binary:
    :param gt_binary:
    :param cast_to_binary:
    :param show:
    :return: the region discovery dice
    """
    from analysis.connected_region2d_and_scale_free_stat import get_connect_region_2d
    shape = np.shape(predicted_binary)
    assert shape == np.shape(gt_binary)
    assert len(shape) == 3
    if cast_to_binary:
        predicted_binary = np.array(predicted_binary > 0.5, 'float32')
        gt_binary = np.array(gt_binary > 0.5, 'float32')
    num_connected_predict = 0
    num_connected_gt = 0
    overlap_count = 0
    for z in range(shape[2]):
        if show and z % 100 == 0:
            print("z at:", z, "total z:", shape[2])
        id_loc_dict_predict = get_connect_region_2d(predicted_binary[:, :, z])
        id_loc_dict_gt = get_connect_region_2d(gt_binary[:, :, z])
        component_count_predict = len(list(id_loc_dict_predict.keys()))
        component_count_gt = len(list(id_loc_dict_gt.keys()))
        num_connected_predict += component_count_predict
        num_connected_gt += component_count_gt
        for key_gt in range(1, component_count_gt + 1):
            discovered = False
            loc_list_gt = id_loc_dict_gt[key_gt]
            for key_predict in range(1, component_count_predict + 1):
                if not discovered:
                    loc_list_predict = id_loc_dict_predict[key_predict]
                    for locations in loc_list_gt:
                        if locations in loc_list_predict:
                            overlap_count += 1
                            discovered = True
                            break
                else:
                    break

        for key_predict in range(1, component_count_predict + 1):
            discovered = False
            loc_list_predict = id_loc_dict_predict[key_predict]
            for key_gt in range(1, component_count_gt + 1):
                if not discovered:
                    loc_list_gt = id_loc_dict_gt[key_gt]
                    for locations in loc_list_predict:
                        if locations in loc_list_gt:
                            overlap_count += 1
                            discovered = True
                            break
                else:
                    break
    if show:
        print("overlap count:", overlap_count)
        print("num connected gt:", num_connected_gt)
        print("num connected predict:", num_connected_predict)
        print("region discovery dice:", overlap_count / (num_connected_gt + num_connected_predict))
    return overlap_count / (num_connected_gt + num_connected_predict)


def region_discovery_dice_3d(predicted_binary, gt_binary, cast_to_binary=True, show=False, recall_and_precision=True,
                             volume_weighted=True):
    """

    Calculate the region discovery dice for 3D arrays
    return: number of connected region overlapped / total number of connected region, ranges in [0, 1]

    :param volume_weighted: False to treat all connected component as equal
    :param recall_and_precision: return recall and precision
    :param predicted_binary:
    :param gt_binary:
    :param cast_to_binary:
    :param show:
    :return: the region discovery dice, or (dice, recall, precision)
    """
    from analysis.connect_region_detect import get_sorted_connected_regions
    shape = np.shape(predicted_binary)
    assert shape == np.shape(gt_binary)
    assert len(shape) == 3 or len(shape) == 2
    if cast_to_binary:
        predicted_binary = np.array(predicted_binary > 0.5, 'float32')
        gt_binary = np.array(gt_binary > 0.5, 'float32')
    id_loc_dict_predict = get_sorted_connected_regions(predicted_binary, threshold=None, show=show)
    id_loc_dict_gt = get_sorted_connected_regions(gt_binary, threshold=None, show=show)
    num_connected_predict = len(id_loc_dict_predict)
    num_connected_gt = len(id_loc_dict_gt)
    overlap_count = 0
    gt_discovered = 0
    predicted_discovered = 0

    overlap_volume = 0
    gt_discovered_volume = 0
    predicted_discovered_volume = 0
    volume_gt = np.sum(gt_binary)
    volume_predicted = np.sum(predicted_binary)
    if volume_gt == 0:
        print("the ground truth is 0")
        if recall_and_precision:
            return np.nan, np.nan, np.nan
        else:
            return np.nan
    if volume_predicted == 0:
        print("the prediction is 0")
        if recall_and_precision:
            return 0, 0, 1
        else:
            return 0

    for key_gt in range(1, num_connected_gt + 1):
        discovered = False
        loc_list_gt = id_loc_dict_gt[key_gt]
        for key_predict in range(1, num_connected_predict + 1):
            if not discovered:
                loc_list_predict = id_loc_dict_predict[key_predict]
                for locations in loc_list_gt:
                    if locations in loc_list_predict:
                        overlap_count += 1
                        overlap_volume += len(loc_list_gt)
                        gt_discovered += 1
                        gt_discovered_volume += len(loc_list_gt)
                        discovered = True
                        break
            else:
                break

    for key_predict in range(1, num_connected_predict + 1):
        discovered = False
        loc_list_predict = id_loc_dict_predict[key_predict]
        for key_gt in range(1, num_connected_gt + 1):
            if not discovered:
                loc_list_gt = id_loc_dict_gt[key_gt]
                for locations in loc_list_predict:
                    if locations in loc_list_gt:
                        overlap_count += 1
                        overlap_volume += len(loc_list_predict)
                        predicted_discovered += 1
                        predicted_discovered_volume += len(loc_list_predict)
                        discovered = True
                        break
            else:
                break

    if show:
        print("overlap count:", overlap_count)
        print("num connected gt:", num_connected_gt)
        print("num connected predict:", num_connected_predict)
        print("num gt region recalled:", gt_discovered)
        print("recall, precision:", gt_discovered / num_connected_gt, predicted_discovered / num_connected_predict)
        print("region discovery dice:", overlap_count / (num_connected_gt + num_connected_predict))

    if not volume_weighted:
        if recall_and_precision:
            return overlap_count / (num_connected_gt + num_connected_predict), gt_discovered / num_connected_gt, \
                   predicted_discovered / num_connected_predict
        return overlap_count / (num_connected_gt + num_connected_predict)
    else:
        if recall_and_precision:
            return overlap_volume / (volume_gt + volume_predicted), gt_discovered_volume / volume_gt, \
                   predicted_discovered_volume / volume_predicted
        return overlap_volume / (volume_gt + volume_predicted)


def roc_auc_score(gt_score, predict_score, show=False, save_path='/Users/richard/Desktop/mac_transfer/AKI_AUC.svg',
                  fpr_at_recall=None, tpr_at_precision=None):
    from sklearn import metrics
    from matplotlib import pyplot as plt

    auc = metrics.roc_auc_score(gt_score, predict_score)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(gt_score, predict_score)

    if fpr_at_recall is not None:
        index = 0
        while not true_positive_rate[index] >= fpr_at_recall:
            index += 1
        print("FRP at recall %0.4f" % fpr_at_recall, false_positive_rate[index])

    if tpr_at_precision is not None:
        index = 0
        while not false_positive_rate[index] >= 1 - tpr_at_precision:
            index += 1
        print("TPR at precision %0.4f" % (1 - tpr_at_precision), true_positive_rate[index])

    plt.figure(figsize=(12, 12), dpi=300)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve for Predicting AKI")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if not show:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    a = Functions.pickle_load_object('/Users/richard/Desktop/mac_transfer/AKI_gt.pickle')
    a = np.array(a, 'float32')
    a = np.array(a > 1.5, 'float32')

    b = Functions.pickle_load_object('/Users/richard/Desktop/mac_transfer/AKI_predict.pickle')
    b = np.array(b, 'float32') / 3

    roc_auc_score(a, b, fpr_at_recall=0.6, tpr_at_precision=0.6)

    exit()

    test_image_gt = np.zeros([20, 20], 'float32')
    test_image_gt[2: 5, 2: 5] = 1
    test_image_gt[16: 18, 16: 18] = 1
    test_image_gt[10: 14, 10: 14] = 1

    test_image_gt[1, 7] = 1



    # Functions.image_show(test_image_gt)

    test_image_predict = np.zeros([20, 20], 'float32')

    test_image_predict[2: 6, 3: 5] = 1
    test_image_predict[15: 18, 16: 19] = 1
    test_image_predict[12: 14, 13: 14] = 1

    print(region_discovery_dice_3d(test_image_predict, test_image_gt, volume_weighted=True))

    exit()
    array = np.load('/home/zhoul0a/Desktop/vein_artery_identification/rescaled_gt/f036_2020-03-10.npz')['array'][:, :, :, 3:5]
    ct = np.load('/home/zhoul0a/Desktop/vein_artery_identification/rescaled_ct/f036_2020-03-10.npy')
    artery = array[:, :, :, 0]
    vein = array[:, :, :, 1]
    import basic_tissue_prediction.predict_rescaled as predictor
    stage_one = predictor.predict_blood_vessel_stage_one_rescaled_array(ct)
    predicted = predictor.get_prediction_blood_vessel(ct, stage_one_array=stage_one)
    raw = np.array(stage_one > 0.76 * np.max(stage_one), 'float32')
    import Tool_Functions.Functions as Functions
    Functions.save_np_array('/home/zhoul0a/Desktop/vein_artery_identification/visualizations/', 'stage_two.npz', predicted, compress=True)
    Functions.save_np_array('/home/zhoul0a/Desktop/vein_artery_identification/visualizations/', 'stage_one.npz',
                            stage_one, compress=True)
    exit()
    print(region_discovery_dice_z_axis(raw, artery + vein, show=True))
    print(region_discovery_dice_z_axis(predicted, artery + vein, show=True))
    exit()
