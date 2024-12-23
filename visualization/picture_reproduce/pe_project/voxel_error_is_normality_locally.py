import numpy as np
import Tool_Functions.Functions as Functions
import Tool_Functions.statistical_tests as tests
import collaborators_package.denoise_chest_ct.denoise_predict as de_noising
from pulmonary_embolism.diagnose.get_expected import get_expected_ct_signal

ct_array = np.load('/media/zhoul0a/New Volume/rescaled_ct_and_semantics/rescaled_ct_float16/'
                   'healthy_people/xwzc/xwzc000014.npz')['array']

ct_array = de_noising.denoise_rescaled_array(ct_array)

blood_mask = np.load(
    '/media/zhoul0a/New Volume/rescaled_ct_and_semantics/semantics/healthy_people/'
    'xwzc/blood_mask/xwzc000014.npz')['array']
model_pe_path = '/home/zhoul0a/Desktop/pulmonary_embolism/check_point_guide/training/' \
                'high_resolution_denoise_include_rad/best_model_guided.pth'
expect_ct, prediction_count_array, package = \
    get_expected_ct_signal(ct_array, high_resolution=True, model_guided_path=model_pe_path,
                           blood_vessel_mask=blood_mask, return_ct_data_package=True)

print(np.max(package[2]))
depth_greater_4 = np.array(package[2] >= (np.max(package[2] - 3)), 'float32')
prediction_count_array = prediction_count_array * depth_greater_4

loc_list_108 = Functions.get_location_list(np.where(prediction_count_array == 108))

sample_dif_list = []

for loc in loc_list_108:
    value_dif = ct_array[loc] - expect_ct[loc]
    sample_dif_list.append(value_dif)

print('there are', len(sample_dif_list))

print(np.mean(sample_dif_list) * 1600, np.std(sample_dif_list) * 1600, np.mean(np.abs(sample_dif_list) * 1600))

tests.normality_test(
    sample_dif_list, show_qq_norm=True,
    save_path='/home/zhoul0a/Desktop/pulmonary_embolism/figures/normality_for_local_error.svg')
