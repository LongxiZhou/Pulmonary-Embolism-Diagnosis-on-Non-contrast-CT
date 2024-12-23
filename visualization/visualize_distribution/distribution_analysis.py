from matplotlib import pyplot as plt
import Tool_Functions.Functions as Functions
import math
import numpy as np
import seaborn as sns


def histogram_list(value_list, interval=None, save_path=None, show=True, range_show=None,
                   x_name='x', y_name='y', title='Histogram Plot'):
    assert len(value_list) > 0
    if interval is None:
        interval = int(math.sqrt(len(value_list))) + 1

    plt.hist(value_list, interval, range=range_show)
    # plt.style.use('seaborn-poster')
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def distribution_plot(value_dict, y_label=None, title=None, save_path=None, y_range_show=None,
                      show_data_points=True, method='box_plot', showfliers=False, dpi=600, nan_policy=None,
                      ylimit=None, palette=None, dot_size=1):
    """

    :param palette:
    :param ylimit: (y_low, y_high)
    :param nan_policy: None for delete nan, float to set nan to this value.
    :param dpi:
    :param showfliers: show outlier
    :param method: 'box_plot', 'violin_plot'
    :param value_dict: {variable_name: [value, ]}
    :param y_label:
    :param title:
    :param save_path:
    :param y_range_show: like (-20, 40)
    :param show_data_points:
    :param dot_size

    :return:
    """

    x_value_list = []
    y_value_list = []

    variable_name_list = list(value_dict.keys())

    for variable_name in variable_name_list:
        value_list_at_this_name = value_dict[variable_name]
        for value in value_list_at_this_name:
            if not value > -np.inf:
                if nan_policy is None:
                    continue
                else:
                    value = nan_policy
            y_value_list.append(value)
            x_value_list.append(variable_name)
    if method == 'violin_plot':
        ax = sns.violinplot(x=x_value_list, y=y_value_list, order=variable_name_list,
                            showfliers=showfliers, palette=palette)
    elif method == 'box_plot':
        ax = sns.boxplot(x=x_value_list, y=y_value_list, order=variable_name_list,
                         showfliers=showfliers, palette=palette)
    else:
        raise ValueError
    if show_data_points:
        # sns.swarmplot(x=x_value_list, y=y_value_list, order=variable_name_list, color=".25")

        # if too much data, use this one
        sns.stripplot(x=x_value_list, y=y_value_list, order=variable_name_list, color=".25", palette=None, size=dot_size)

    if y_range_show is not None:
        ax.set_ylim(y_range_show)
    ax.set_ylabel(y_label, fontsize=11)
    plt.xticks(fontsize=10)
    ax.set_title(title, fontsize=12)

    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi
    if ylimit is not None:
        plt.ylim(ylimit)
    if save_path is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 12)
        plt.savefig(save_path)
    else:
        plt.show()


def re_sample_temp(x_value_list, y_value_list, num_out):
    import random
    random.seed(0)

    paired_list = list(zip(x_value_list, y_value_list))

    def compare_func(a, b):
        if a[1] < b[1]:
            return 1
        return -1

    sorted_list = Functions.customized_sort(paired_list, compare_func)

    index_list = list(range(0, len(sorted_list)))

    weight_list = list(range(1, len(sorted_list + 1)))
    weight_list.reverse()
    weight_list = [float(i) / sum(weight_list) for i in weight_list]

    selected_index = np.random.choice(index_list, size=num_out, replace=False, p=weight_list)
    new_list_out = []
    remain_list = []
    for i in range(len(sorted_list)):
        if i in selected_index:
            new_list_out.append(sorted_list[i])
        else:
            remain_list.append(sorted_list[i])


if __name__ == '__main__':

    import os
    top_dict = '/data_disk/pulmonary_embolism_final/statistics/with_gt/clot_av_rad'
    fn_list = os.listdir(top_dict)

    data_list = []
    for fn in fn_list:
        pickle_data = Functions.pickle_load_object(os.path.join(top_dict, fn))
        data_list.append(pickle_data['v1'][0])
    data_list.sort()

    data_list = data_list[0: -20]

    print(data_list[int(0.97 * len(data_list))])

    histogram_list(data_list, interval=100)
    exit()
    actual_clot_volume_list = Functions.pickle_load_object('/data_disk/pulmonary_embolism/pickle_objects/'
                                                           'actual_clot_volume_list.pickle')
    print(len(actual_clot_volume_list))
    new_list = []
    for value in actual_clot_volume_list:
        if value > 100:
            new_list.append(value)
    print(len(new_list))
    new_list = np.sqrt(new_list)
    histogram_list(new_list, 10)
