from plotnine import ggplot, aes, geom_point, geom_smooth
import Tool_Functions.Functions as Functions
import pandas as pd
import numpy as np
import random
import os


def create_line(x_value_list, y_value_list, x_name='x_value', y_name='y_value', save_path=None, max_point=30000,
                show=True):

    if max(x_value_list) == min(x_value_list):
        print(x_name, 'with only one value:', max(x_value_list), 'cannot plot line')
        return None

    num_points = len(x_value_list)
    if num_points > max_point:  # cannot plot too much points
        index_list = np.arange(0, num_points)
        random.shuffle(index_list)

        new_x_value_list = []
        new_y_value_list = []

        for index in range(0, max_point):
            new_x_value_list.append(x_value_list[index_list[index]])
            new_y_value_list.append(y_value_list[index_list[index]])

        x_value_list = new_x_value_list
        y_value_list = new_y_value_list

    # ggplot cannot plot too much points
    data = {x_name: x_value_list,
            y_name: y_value_list,
            "color": []}

    # here can assign color, see https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/
    for i in range(len(data[x_name])):
        data["color"].append('PI')

    # load data into a DataFrame object:
    df = pd.DataFrame(data)

    # geom_point() plot out the scatter voxels.  geom_smooth() gives the trend line
    image = ggplot(df, aes(x=x_name, y=y_name)) + geom_smooth(method="loess")  # + geom_point()
    image.draw(show=show)

    # save image to certain filename and directory
    if save_path is not None:
        father_dict = Functions.get_father_dict(save_path)
        if not os.path.exists(father_dict):
            os.makedirs(father_dict)
        file_name = save_path[len(father_dict) + 1:]
        image.save(filename=file_name, path=father_dict, dpi=1200)


if __name__ == '__main__':
    create_line(Functions.pickle_load_object('/home/zhoul0a/Desktop/transfer/CS300/branch_list.pickle'),
                Functions.pickle_load_object('/home/zhoul0a/Desktop/transfer/CS300/value_list.pickle'),
                max_point=10000,
                save_path='/home/zhoul0a/Desktop/transfer/CS300/trend_temp.svg')
