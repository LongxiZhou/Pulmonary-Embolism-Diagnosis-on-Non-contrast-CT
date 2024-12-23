import visualization.visualize_distribution.distribution_analysis as distribution
import Tool_Functions.Functions as Functions
import numpy as np

value_dict = Functions.pickle_load_object(
    '/data_disk/pulmonary_embolism_final/segmentation_performance/overall_performance_trim_4000_augment.pickle')

save_path = '/data_disk/pulmonary_embolism_final/pictures/augment_seg_dice.svg'

for key, value in value_dict.items():
    print(key)
    print(value)

dice_list_each_case = value_dict['dice_list_each_case']
dice_list_each_case_new = []
for dice in dice_list_each_case:
    if not dice > -np.inf:
        dice = 0
        continue
    dice_list_each_case_new.append(dice)

print("dice average", np.average(dice_list_each_case_new),
      "dice std", np.std(dice_list_each_case_new))

rd_dice_list_each_case = value_dict['region_discovery_dice_list_each_case']
rd_dice_list_each_case_new = []
for dice in rd_dice_list_each_case:
    if not dice > -np.inf:
        dice = 0
        continue
    rd_dice_list_each_case_new.append(dice)

print("region_discovery_dice average", np.average(rd_dice_list_each_case_new),
      "region_discovery_dice std", np.std(rd_dice_list_each_case_new))

new_value_dict = {'Dice Each Patient': value_dict['dice_list_each_case'],
                  'Region Discovery Dice Each Patient': value_dict['region_discovery_dice_list_each_case']}

distribution.distribution_plot(new_value_dict, nan_policy=None,
                               save_path=save_path,
                               title='Clot Segmentation Dice on Non-Contrast CT')
