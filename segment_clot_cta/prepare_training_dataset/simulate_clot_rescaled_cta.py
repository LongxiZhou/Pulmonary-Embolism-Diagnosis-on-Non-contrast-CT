import Tool_Functions.Functions as Functions
import numpy as np
from functools import partial
from Tool_Functions.fitting_and_simulation import cubic_spline_interpolation
import random


"""
each clot_sample is a dict:
clot_sample_dict: a dict, with key "loc_depth_set" and "range_clot"
clot_sample_dict["loc_clot_set"] = {(x, y, z), }
clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, ..., 'max_depth': max_depth}  here b is the clot depth
the mass center for the location x, y, z is (0, 0, 0)
clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations
"""


def apply_clot_on_rescaled_cta(rescaled_cta, vessel_mask, clot_sample_dict, func_reset_ct=None, clot_center_loc=None,
                               noise_cta=None, show_func_reset=False, global_bias=0.):
    """
    :param global_bias: the clot region is added a global shift to reduce difficulty. in rescaled, i.e., HU value / 1600
    :param show_func_reset:
    :param rescaled_cta:
    :param vessel_mask: blood vessel mask or blood region mask, binary, numpy float32
    :param clot_sample_dict:
    :param func_reset_ct: new_ct_value = func_reset_ct(depth), here depth is int, new_ct_value is rescaled
    :param clot_center_loc: the mass center location for the clot, in (x, y, z)
    :param noise_cta: the noise level for this CTA, in rescaled, i.e., HU value / 1600
    :return: loc_array, new_ct_value

    rescaled_cta_with_clot: rescaled_cta[loc_array] = new_ct_value

    """
    shape = np.shape(rescaled_cta)

    if clot_center_loc is None:
        # random select a location in the vessel for the mass center of clot
        loc_array_vessel_mask = np.where(vessel_mask > 0.5)
        num_point = len(loc_array_vessel_mask[0])
        select_id = random.randint(0, num_point - 1)
        clot_center_loc = (loc_array_vessel_mask[0][select_id],
                           loc_array_vessel_mask[1][select_id], loc_array_vessel_mask[2][select_id])

    # form clot_loc_array
    loc_list_clot = []
    loc_list_original = []
    for loc in list(clot_sample_dict["loc_clot_set"]):  # here we discard loc outside the vessel mask.
        new_loc = (int(loc[0] + clot_center_loc[0]), int(loc[1] + clot_center_loc[1]), int(loc[2] + clot_center_loc[2]))
        if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[2] < 0:
            continue
        if new_loc[0] >= shape[0] or new_loc[1] >= shape[1] or new_loc[2] >= shape[2]:
            continue
        if vessel_mask[new_loc] > 0.5:
            loc_list_clot.append(new_loc)
            loc_list_original.append(loc)
    clot_loc_array = Functions.get_location_array(loc_list_clot)

    max_clot_depth = clot_sample_dict["clot_depth_dict"]["max_depth"]
    # we found for each PE patient, when select the largest clot for this patient
    # the max depth for the clot ranges from 5-16

    if func_reset_ct is None:
        # here we establish the default func_reset_ct
        value_array = rescaled_cta[clot_loc_array]
        func_reset_ct = get_default_func_reset_ct(value_array, max_clot_depth, noise_cta, show=show_func_reset,
                                                  global_bias=global_bias)

    for index in range(len(loc_list_clot)):
        depth_index = clot_sample_dict["clot_depth_dict"][loc_list_original[index]]
        rescaled_cta[loc_list_clot[index]] = func_reset_ct(depth_index)

    return clot_loc_array, rescaled_cta[clot_loc_array]


def get_default_func_reset_ct(value_array, max_clot_depth, noise_cta=None, show=False, global_bias=0.):
    """
    :param global_bias: the clot region is added a global shift to reduce difficulty. in rescaled, i.e., HU value / 1600
    :param show:
    :param noise_cta: the noise level for this CTA, in rescaled, i.e., HU value / 1600
    :param max_clot_depth: the max depth for the clot to be apply
    :param value_array: value_array = rescaled_cta[clot_loc_array]
    :return: func_reset_ct
    """
    # here we establish the default func_reset_ct
    if len(value_array) == 0:
        return None
    median_blood_ct = np.median(value_array)  # the CT value for clot_depth 0
    median_blood_ct = Functions.change_to_HU(median_blood_ct)
    if noise_cta is None:
        # I planned to use the std in clot region to estimate the noise, but seems is to large,
        # noise_cta = min(np.std(value_array), 150 / 1600)
        noise_cta = 0  # 20 / 1600  # noise for regular CTA is around 20 HU
    clot_ct_depth_2 = random.randint(-50, 100)  # average signal for clot depth 20%
    clot_ct_depth_5 = random.randint(-50, 50) + clot_ct_depth_2
    # clot_ct_depth_5 = max(-100, clot_ct_depth_5)
    clot_ct_depth_10 = random.randint(-50, 50) + clot_ct_depth_5
    clot_ct_depth_10 = max(-100, clot_ct_depth_10)

    x_value_list = [0, max_clot_depth / 5, max_clot_depth / 2, max_clot_depth]
    y_value_list = [median_blood_ct, clot_ct_depth_2, clot_ct_depth_5, clot_ct_depth_10]
    y_value_list = Functions.change_to_rescaled(np.array(y_value_list, 'float32'))
    y_value_list[1::] = y_value_list[1::] + global_bias

    if show:
        print("median_blood_ct", median_blood_ct)
        print(Functions.change_to_HU(np.array(y_value_list)))
        cubic_spline_interpolation(x_value_list, Functions.change_to_HU(np.array(y_value_list)),
                                   boundary_condition='natural', show=True)

    depth_ct_f = cubic_spline_interpolation(x_value_list, y_value_list, boundary_condition='natural', show=False)
    # set show=True to see the curve for depth_ct_f

    func_reset_ct = partial(func_reset_ct_template, depth_ct_f=depth_ct_f, noise_hu=noise_cta)

    return func_reset_ct


def func_reset_ct_template(depth, depth_ct_f, noise_hu):
    return depth_ct_f(depth) + random.uniform(-noise_hu / 2, noise_hu / 2)


if __name__ == '__main__':

    cta = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/non_PE_CTA/rescaled_ct-denoise/'
                  'AL00013.npz')['array']

    vessel_mask_ = np.load('/data_disk/pulmonary_embolism/segment_clot_on_CTA/non_PE_CTA/secondary_semantics/'
                           'blood_region_strict/AL00013.npz')['array']
    # import visualization.visualize_3d.visualize_stl as stl
    # from smooth_mask.get_lung_vessel_blood_region.inference import get_blood_region_rescaled_mask
    # print(np.sum(vessel_mask_))
    # stl.visualize_numpy_as_stl(vessel_mask_)
    # vessel_mask_ = get_blood_region_rescaled_mask(vessel_mask_, get_connect=True)
    # print(np.sum(vessel_mask_))
    # stl.visualize_numpy_as_stl(vessel_mask_)

    clot_sample_dict_list = Functions.pickle_load_object('/data_disk/pulmonary_embolism/simulated_lesions/'
                                                         'clot_sample_list_reduced/volume_range_5%/'
                                                         'raw_lesion/10000_to_25000.pickle')
    select_clot = clot_sample_dict_list[10]
    print(select_clot["clot_depth_dict"]["max_depth"])

    clot_loc_array_, value_array_ = apply_clot_on_rescaled_cta(cta, vessel_mask_, select_clot, noise_cta=50 / 1600,
                                                               show_func_reset=True, global_bias=-0.1)

    cta[clot_loc_array_] = value_array_

    clot_mask = np.zeros([512, 512, 512], 'float32')
    clot_mask[clot_loc_array_] = 1

    z_list = list(set(clot_loc_array_[2]))
    z_list.sort()

    cta = np.clip(cta, Functions.change_to_rescaled(-600), Functions.change_to_rescaled(300))

    for z in z_list[::4]:
        Functions.merge_image_with_mask(cta[:, :, z], clot_mask[:, :, z])
