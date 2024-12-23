import math
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats


def t_test(list_a, list_b, equal_std=False, dependent=False):
    """
    compare whether the mean of two normal distribution is different
    :param list_a: sampled values from distribution a
    :param list_b: sampled values from distribution a
    :param equal_std: whether the distribution is known to have the same std
    :param dependent: in [True, False, "unknown"]
    :return:
    """
    print("sample_num for two distribution:", len(list_a), len(list_b))
    print("distribution assumption:", "normal distribution", "equal_std:", equal_std, "dependency:", dependent)
    print(stats.ttest_ind(list_a, list_b))


def chi2_contigency_test(list_a, list_b, a_level=3, b_level=3):
    """
    two different variables forms two lists: list_a, list_b. test whether they are independent.
    each index is coupled, (list_a[0], list_b[0]) ..., whether values in list_a will influence values in list_b
    :param list_a: sampled values from variable a
    :param list_b: sampled values from variable b
    :param a_level: level of variable a
    :param b_level: level of variable b
    :return: p value
    """
    from scipy.stats import chi2_contingency

    length_a = len(list_a)
    length_b = len(list_b)
    assert length_a == length_b
    interval_a = round(length_a / a_level)
    interval_b = round(length_b / b_level)

    p_value_log = 0

    tested_num = 0
    potential_p = []

    for j in range(1000):
        list_a_new = list(list_a)
        list_b_new = list(list_b)
        for i in range(length_a):  # add a small noise to make every observation distinguishable
            list_a_new[i] = list_a[i] + random.random() / 10000000
            list_b_new[i] = list_b[i] + random.random() / 10000000
        sorted_a = list(list_a_new)
        sorted_a.sort()
        sorted_b = list(list_b_new)
        sorted_b.sort()

        contigency_array = np.zeros([a_level, b_level], 'int32')
        for i in range(length_a):  # patient i
            value_a = list_a_new[i]
            value_b = list_b_new[i]
            loc_a = sorted_a.index(value_a)
            loc_b = sorted_b.index(value_b)
            contigency_array[min(int(loc_a / interval_a), a_level - 1), min(int(loc_b / interval_b), b_level - 1)] += 1
        current_p_log = math.log(chi2_contingency(contigency_array)[1])
        p_value_log = p_value_log + current_p_log
        tested_num += 1
        if current_p_log not in potential_p:
            potential_p.append(p_value_log)
        if tested_num % 100 == 9:
            if np.std(potential_p) / tested_num < 0.01:
                # print("converged at", tested_num - 8)
                break

    p_value_log = p_value_log / tested_num

    return math.exp(p_value_log)


def geometric_mean(inputs_array):  # all_file inputs should greater than 0
    log_out = np.sum(np.log(inputs_array))
    shape = np.shape(inputs_array)
    total_count = 1
    for i in shape:
        total_count *= i
    return math.exp(log_out / total_count)


def dependency_test(list_a, list_b, a_level_trial=(2, 4), b_level_trial=(2, 4), single_value=False):
    a_max = min(a_level_trial[1], len(set(list_a)))
    a_min = min(a_level_trial[0], len(set(list_a)))
    b_max = min(b_level_trial[1], len(set(list_b)))
    b_min = min(b_level_trial[0], len(set(list_b)))
    p_array = np.zeros([a_max - a_min + 1, b_max - b_min + 1], 'float32')
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            p_array[a - a_min, b - b_min] = chi2_contigency_test(list_a, list_b, a, b)
    if single_value:
        return geometric_mean(p_array)
    return p_array


def probability_binomial(n, m):
    if n < 100:
        return math.factorial(n) / math.factorial(m) / math.factorial(n - m) * math.pow(0.5, n)
    log_n_factorial = log_factorial(n)
    if m > 100:
        log_m_factorial = log_factorial(m)
    else:
        log_m_factorial = math.log(math.factorial(m))
    if n - m > 100:
        log_n_m_factorial = log_factorial(n - m)
    else:
        log_n_m_factorial = math.log(math.factorial(n - m))
    return math.exp(log_n_factorial + n * math.log(0.5) - log_m_factorial - log_n_m_factorial)


def log_factorial(n):
    return n * math.log(n) - n + 0.5 * math.log(
        2 * 3.1415926535897932384626433 * n) + 1 / 12 / n - 1 / 360 / n / n / n + 1 / 1260 / n / n / n / n / n - 1 / \
           1680 / n / n / n / n / n / n / n


def normality_test(list_like, show_qq_norm=False, method='ks_test', normalize=True, show=True, save_path=None):
    """

    :param save_path:
    :param show:
    :param list_like: stores the sampling of the variable
    :param show_qq_norm:
    :param method:
    'ks_test', for non-int only and can handle large data number

    :param normalize: normalize to mean 0 and std 1
    :return: statics, a float, p-value (if the list_like is sampled from normal distribution), a float
    """

    list_like = np.array(list_like, 'float32')

    std_for_list_like = np.std(list_like)
    assert std_for_list_like > 0

    if normalize:
        list_like = (list_like - np.mean(list_like)) / std_for_list_like

    if show_qq_norm:
        from statsmodels.api import qqplot
        # create Q-Q plot with 45-degree line added to plot
        if show:
            print('Q-Q plot')
        qqplot(list_like, line='45')
        plt.show()
    if save_path is not None:
        from statsmodels.api import qqplot
        qqplot(list_like, line='45')
        plt.savefig(save_path)

    assert method in ['ks_test', 'shapiro']

    if method == 'ks_test':
        static, p_value = stats.kstest(list_like, 'norm')
        if show:
            print('Using ks test, static:', static, 'p_value:', p_value)
        return static, p_value
    if method == 'shapiro':
        static, p_value = stats.shapiro(list_like)
        if show:
            print('Using shapiro test, static:', static, 'p_value:', p_value)
        return static, p_value

    return None


if __name__ == '__main__':
    test_list = np.random.normal(size=(1000,), loc=4, scale=100)
    a, p = normality_test(test_list, save_path='/home/zhoul0a/Desktop/pulmonary_embolism/figures/normality_for_local_error.svg')
    print(a, p)
