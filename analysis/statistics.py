import scipy
import random
import numpy as np
import math


def chi2_contigency_test(list_a, list_b, a_level=3, b_level=3):
    """
    we have two different variables: list_a, list_b. test whether they are independent.
    :param list_a: variable a
    :param list_b: variable b

    we need to cast a and b into several intervals, and count the frequency of each interval
    :param a_level: level of variable a, i.e., interval of a
    :param b_level: level of variable b, i.e., interval of b
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
    """

    see function chi2_contigency_test

    :param list_a:
    :param list_b:
    :param a_level_trial: use multiple interval for a
    :param b_level_trial: use multiple interval for b
    :param single_value:
    :return:
    """
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
        2 * 3.1415926535897932384626433 * n) + 1 / 12 / n - 1 / 360 / n / n / n + \
           1 / 1260 / n / n / n / n / n - 1 / 1680 / n / n / n / n / n / n / n


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import t, levene
    import matplotlib.pyplot as plt

    x1 = 0.801
    x2 = 0.833
    n1 = 264
    n2 = 264
    std1 = 0.224
    std2 = 0.183

    temp_list_1 = np.random.normal(x1, std1, n1)
    temp_list_1 = np.sqrt(std1 / np.std(temp_list_1)) * temp_list_1
    temp_list_2 = np.random.normal(x2, std2, n2)
    temp_list_2 = np.sqrt(std2 / np.std(temp_list_2)) * temp_list_2
    print(levene(temp_list_1, temp_list_2))

    t_value = abs(x1 - x2) / math.sqrt(std1**2/n1 + std2**2/n2)
    print(t.cdf(-t_value, df=n1 + n2 - 1))
    print((t.logcdf(-t_value, df=n1 + n2 - 1) + math.log(2)) / math.log(10))
    exit()

    fig, ax = plt.subplots(1, 1)
    df = 20
    mean, var, skew, kurt = t.stats(df, moments='mvsk')

    x = np.linspace(t.ppf(0.0001, df),
                    t.ppf(0.9999, df), 100)
    ax.plot(x, t.cdf(x, df),
            'r-', lw=5, alpha=0.6, label='t pdf')

    plt.show()
