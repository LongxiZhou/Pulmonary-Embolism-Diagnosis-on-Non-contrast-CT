import numpy as np
import random
import Tool_Functions.Functions as Functions
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def cubic_spline_interpolation(x_value_list, y_value_list, show=True, boundary_condition='natural'):
    """

    fitting a function: y = f(x) based on given (x_i, y_i)

    :param x_value_list: (x1, x2, ...)
    :param y_value_list: (y1, y2, ...)
    :param show:
    :param boundary_condition: see document for CubicSpline
    :return: a CubicSpline object, denote as f

    value_list = f(input_list, extrapolate=None)
    extrapolate is import when inference values:
    in {'periodic', None}. If 'periodic', periodic extrapolation is used. None for simple extend

    Use f.derivative(nu) to do integral and derivative. nu is times for derivatives, like nu=-1 is one time integral
    f_new = f.derivative(nu)

    float = f.integrate(a, b, extrapolate=None)

    array = f.solve(y)  # get array of x from small to large, so that f(x) = y
    """

    # use bc_type = 'natural' adds the constraints as we described above
    f = CubicSpline(x_value_list, y_value_list, bc_type=boundary_condition)
    if show:
        show_f(f, x_value_list, y_value_list)

    return f


def show_f(f, x_value_list=None, y_value_list=None, x_name='x', y_name='y', title='Cubic Spline Interpolation',
           save_path=None, show=True):
    if x_value_list is None:
        x_value_list = f.x
    if y_value_list is None:
        y_value_list = f(x_value_list)
    x_new = np.linspace(min(x_value_list), max(x_value_list), 100)
    y_new = f(x_new)
    plt.style.use('seaborn-poster')
    plt.figure(figsize=(10, 8))
    plt.plot(x_new, y_new, 'b')
    plt.plot(x_value_list, y_value_list, 'ro')
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def sort_paired_list(x_value_list, y_value_list):
    """

    :param x_value_list:
    :param y_value_list:
    :return: x_value_list_sorted (from small to large), y_value_list_sorted
    """
    locations = Functions.get_location_list([x_value_list, y_value_list])

    def func_compare(item_a, item_b):
        if item_a[0] > item_b[0]:
            return 1
        return -1

    loc_array = Functions.get_location_array(Functions.customized_sort(locations, func_compare, reverse=False),
                                             dtype='float32')

    return list(loc_array[0]), list(loc_array[1])


def form_probability_distribution_continues_from_value_frequency(x_value_list, frequency_list, show=True, ):
    """

    probability distribution must be continues.

    :param x_value_list: sample points for a distribution
    :param frequency_list: relative frequency or count for given x, will automatically do normalization
    :param show:
    :return: pdf, cdf
    """
    assert len(x_value_list) == len(frequency_list)
    x_value_list, frequency_list = sort_paired_list(x_value_list, frequency_list)

    num_intervals = len(x_value_list) - 1
    max_x, min_x = max(x_value_list), min(x_value_list)
    average_interval = (max_x - min_x) / num_intervals

    new_x_value_list = [min_x - average_interval, ] + list(x_value_list) + [max_x + average_interval, ]
    new_y_value_list = [0, ] + list(frequency_list) + [0, ]
    assert min(frequency_list) >= 0

    start = 0
    while new_y_value_list[start] == 0 and new_y_value_list[start + 1] == 0:
        start += 1

    end = len(new_y_value_list) - 1
    while new_y_value_list[end] == 0 and new_y_value_list[end - 1] == 0:
        end -= 1

    new_x_value_list = new_x_value_list[start: end + 1]
    new_y_value_list = new_y_value_list[start: end + 1]

    f = cubic_spline_interpolation(new_x_value_list, new_y_value_list, show=False, boundary_condition='natural')
    cdf = f.derivative(-1)
    max_cdf = cdf(new_x_value_list[-1])

    if show:
        print("normalized probability distribution:")
    normalized_f = cubic_spline_interpolation(new_x_value_list, np.array(new_y_value_list, 'float64') / max_cdf,
                                              show=show, boundary_condition='natural')
    normalized_cdf = normalized_f.derivative(-1)

    if show:
        print("cumulative distribution function:")
        show_f(normalized_cdf, new_x_value_list, None)

    return normalized_f, normalized_cdf


def form_probability_distribution_continues_from_value_list(observation_value_list, trim_range=(-np.inf, np.inf),
                                                            num_interval=None, show=True, method_get_interval=np.log):

    observation_value_list = Functions.deep_copy(observation_value_list)
    new_list = []
    for value in observation_value_list:
        if value < trim_range[0] or value > trim_range[1]:
            continue
        new_list.append(value)
    observation_value_list = new_list

    if num_interval is None:
        num_interval = int(method_get_interval(len(observation_value_list))) + 1
    min_value, max_value = np.min(observation_value_list), np.max(observation_value_list)
    interval_value = (max_value - min_value) / num_interval

    frequency_list = np.zeros((num_interval, ), 'float32')
    slot_value_dict = {}
    for slot in range(0, num_interval):
        slot_value_dict[slot] = []

    for value in observation_value_list:
        slot = int((value - min_value) / interval_value)
        if slot == num_interval:
            slot = slot - 1
        frequency_list[slot] += 1
        slot_value_dict[slot].append(value)

    x_value_list = []
    for slot in range(0, num_interval):
        if len(slot_value_dict[slot]) == 0:
            x_value_list.append(min_value + (slot + 0.5) * interval_value)
            continue
        x_value_list.append(np.average(slot_value_dict[slot]))

    return form_probability_distribution_continues_from_value_frequency(x_value_list, frequency_list, show=show)


def get_random_variable_from_cdf(cdf, num_value_selected=1):
    """

    :param cdf: a CubicSpline object
    :param num_value_selected:
    :return: list of selected value. [x1, x2, x3, ...]
    """
    x_value_list = cdf.x
    x_min, x_max = x_value_list[0], x_value_list[-1]
    max_cdf = cdf(x_value_list[-1])

    percentile_list = np.random.uniform(0, max_cdf, size=(num_value_selected,))

    def get_valid_root(root_list):
        for root in root_list:
            if x_min <= root <= x_max:
                return root
        raise ValueError("no root in the range of x, please check CDF and percentile_list")

    list_selected_variable = []
    for percentile in percentile_list:
        list_selected_variable.append(get_valid_root(cdf.solve(percentile)))

    return list_selected_variable


def get_random_variable_from_ppf(ppf, num_value_selected=1):
    """

    :param ppf: ppf is the inverse for cdf, like scipy.stats.norm.ppf
    :param num_value_selected:
    :return: list of selected value. [x1, x2, x3, ...]
    """

    percentile_list = np.random.uniform(0, 1, size=(num_value_selected,))

    list_selected_variable = []
    for percentile in percentile_list:
        list_selected_variable.append(ppf(percentile))

    return list_selected_variable


def up_sample_linearly_paired_samples(x_list, y_list, new_sample_number=0, selected_num_for_new_point=2, y_error=0,
                                      x_bound=None, selected_x_interval=np.inf):
    assert len(x_list) == len(y_list)

    x_list_new = list(x_list)
    y_list_new = list(y_list)

    y_max, y_min = np.max(y_list_new), np.min(y_list_new)

    y_error = y_error * selected_num_for_new_point

    length = len(x_list)

    count = 0
    while count < new_sample_number:
        x_temp = []
        y_temp = []
        weight_list = []

        qualified = False
        while not qualified:
            for j in range(selected_num_for_new_point):
                weight_list.append(random.uniform(0, 1))
                index_select = random.randint(0, length - 1)
                x_temp.append(x_list[index_select])
                y_s = y_list[index_select] + random.uniform(-y_error, y_error)
                while not y_min < y_s < y_max:
                    y_s = y_list[index_select] + random.uniform(-y_error, y_error)
                y_temp.append(y_s)
            if not (np.max(x_temp) - np.min(x_temp)) < selected_x_interval:
                x_temp = []
                y_temp = []
                weight_list = []
            if len(x_temp) > 0:
                qualified = True

        weight_list = np.array(weight_list)
        weight_list = weight_list / np.sum(weight_list)
        x_new = 0
        y_new = 0
        for j in range(selected_num_for_new_point):
            x_new = weight_list[j] * x_temp[j]
            y_new = weight_list[j] * y_temp[j]

        if x_bound is not None:
            if not x_new > x_bound[0] and x_new < x_bound[1]:
                continue

        x_list_new.append(x_new)
        y_list_new.append(y_new)
        count += 1

    return x_list_new, y_list_new


if __name__ == '__main__':

    pdf_, cdf_ = form_probability_distribution_continues_from_value_frequency([0, 1, 2, 4, 5, 6, 7], [0, 0, 3, 2, 1, 0, 0], show=False)
    show_f(pdf_)
    show_f(cdf_)

    value_list = get_random_variable_from_cdf(cdf_, 100000)

    from visualization.visualize_distribution.distribution_analysis import histogram_list

    histogram_list(value_list, interval=20)


