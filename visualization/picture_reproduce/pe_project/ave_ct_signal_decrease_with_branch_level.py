from plotnine import ggplot, aes, geom_point, geom_smooth
import Tool_Functions.Functions as Functions
import pandas as pd


############################################
# get the CT value and branch-level pairs
############################################
def get_ct_value_branch_pair(name_id):
    """

    :param name_id:
    :return: branch_list, value_list,

    (branch_list[i], value_list[i]) is the (branch_level, ct_value_in_hu) pair, which forms a data point

    """
    import numpy as np
    import os

    fn_list = os.listdir('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise')
    array = np.load('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise/' + fn_list[name_id])['array']

    # from collaborators_package.ct_volumetric_effect.direct_de_blur import deblur_ct
    # array = deblur_ct(array, z_min=100, z_max=400)

    array = Functions.change_to_HU(array)
    blood_region_strict_mask = np.load('/data_disk/RAD-ChestCT_dataset/secondary_semantics/blood_region_strict/'
                                       + fn_list[name_id])['array']
    blood_branch_map = np.load('/data_disk/RAD-ChestCT_dataset/secondary_semantics/blood_branch_map/'
                               + fn_list[name_id])['array']
    blood_branch_map = blood_branch_map * np.array(array > -250, 'float32') * blood_region_strict_mask

    loc_list_blood = Functions.get_location_list(np.where(blood_branch_map > 0))

    def sort_func(loc_a, loc_b):
        if blood_branch_map[loc_a] > blood_branch_map[loc_b]:
            return 1
        return -1

    Functions.customized_sort(loc_list_blood, compare_func=sort_func)

    value_list = []
    branch_list = []
    for loc in loc_list_blood:
        if blood_branch_map[loc] > 7:
            continue
        value_list.append(array[loc])
        branch_list.append(blood_branch_map[loc])

    # Functions.show_data_points(branch_list, value_list)

    Functions.pickle_save_object('/home/zhoul0a/Desktop/transfer/CS300/branch_list.pickle', branch_list)
    Functions.pickle_save_object('/home/zhoul0a/Desktop/transfer/CS300/value_list.pickle', value_list)

    return branch_list, value_list


get_ct_value_branch_pair(3)


# ggplot cannot plot too much points
data = {"branch_level": Functions.pickle_load_object('/home/zhoul0a/Desktop/transfer/CS300/branch_list.pickle')[0::4],
        "CT value (HU)": Functions.pickle_load_object('/home/zhoul0a/Desktop/transfer/CS300/value_list.pickle')[0::4],
        "color": []}


# here can assign color, see https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/
for i in range(len(data["branch_level"])):
    data["color"].append('PI')

# load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)

# geom_point() plot out the scatter voxels.  geom_smooth() gives the trend line
image = ggplot(df, aes(x="branch_level", y="CT value (HU)")) + geom_smooth(method="loess")  # + geom_point()
image.draw(show=True)

# save image to certain filename and directory
image.save(filename='trend.svg', path='/home/zhoul0a/Desktop/transfer/CS300/', dpi=1200)

