"""
Background:

even with perfect registration, the registered vessel and fixed vessel will not have dice = 1,
because there are segmentation error for vessels.

the registration will align lung mask, big vessels satisfactorily.

registration will make the segmentation for vessel more difficult.


Solution:
1) normalize the non and CTA, and get flow from CTA to non, and non to CTA.
2) smooth the flow, and divide the flows by half, then apply corresponding flow to CTA_simulated_non and apply to non.
This step, register CTA_simulated_non and non in the middle state, denote as CTA_hr, non_hr
3) lung mask and blood vessel outside lung is the union of CTA_hr and non_hr.
lung mask will be used to segment vessels
inside lung, vessel outside lung will be used to add landmark for registration (after segment vessel)
4) get vessel probability inside lung, with predict_rescaled.get_prediction_blood_vessel, for CTA_hr and non_hr
5) register CTA_hr to non_hr, based non vessel inside lung. get CTA_hrr
6) register CTA_hrr to non
"""
import os
import pe_dataset_management.registration.register_cta_to_ct.register_in_pe_paired_database as register_cta_to_non
import pe_dataset_management.registration.register_ct_to_cta.register_in_pe_paired_database as register_non_to_cta
import pe_dataset_management.basic_functions as basic_functions
from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array import \
    blur_flow_with_convolution_kernel
from pe_dataset_management.inference_general_registration import register_with_given_flow

import format_convert.basic_transformations as basic_transform
import analysis.get_surface_rim_adjacent_mean as get_surface
import analysis.center_line_and_depth_3D as get_depth
import numpy as np
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import format_convert.spatial_normalize as spatial_normalize

from pe_dataset_management.inference_general_registration import normalization, register_with_given_flow, \
    register

from pe_dataset_management.registration.super_accurate_vessel_registration.smooth_flow_and_array import \
    smooth_256_then_up_sample_to_512


def register_and_smooth_flow(non_hr, cta_hr, vessel_seg_non_hr, vessel_seg_cta_hr):
    """

    :param non_hr:
    :param cta_hr:
    :param vessel_seg_non_hr:
    :param vessel_seg_cta_hr:
    :return:
    """

    registered_cta, registered_vessel_cta, fixed_non, fixed_vessel_non, register_flow = \
        register(cta_hr, non_hr, vessel_seg_cta_hr, vessel_seg_non_hr, two_stage=False,
                 down_sample=True, return_flow=True)

    print("original dice", metrics.dice_score_two_class(vessel_seg_cta_hr, vessel_seg_non_hr))
    print("registered_dice", metrics.dice_score_two_class(registered_vessel_cta, fixed_vessel_non))

    register_flow_smoothed_512 = smooth_256_then_up_sample_to_512(register_flow, show_jacobi=True)

    vessel_cta_with_smooth_flow = register_with_given_flow(vessel_seg_cta_hr, register_flow_smoothed_512)
    vessel_cta_with_smooth_flow = np.array(vessel_cta_with_smooth_flow > 0.5, 'float32')
    vessel_cta_with_smooth_flow = vessel_cta_with_smooth_flow + get_surface.get_surface(
        vessel_cta_with_smooth_flow, outer=True, strict=False)
    vessel_cta_with_smooth_flow = vessel_cta_with_smooth_flow - get_surface.get_surface(
        vessel_cta_with_smooth_flow, outer=False, strict=False)
    cta_with_smooth_flow = register_with_given_flow(cta_hr, register_flow_smoothed_512)

    Functions.save_np_array(
        top_dict + '/step_' + str(current_step + 1), 'cta_hr.npy',
        cta_with_smooth_flow, compress=False)
    Functions.save_np_array(
        top_dict + '/step_' + str(current_step + 1),
        'vessel_seg_cta_hr.npz',
        vessel_cta_with_smooth_flow, compress=True)

    print("registered_dice_with smooth flow",
          metrics.dice_score_two_class(vessel_cta_with_smooth_flow, vessel_seg_non_hr))

    image_up = Functions.merge_image_with_mask(
        np.clip(non_hr[:, :, 256], -0.25, 0.7), vessel_seg_non_hr[:, :, 256], show=False)
    image_down = Functions.merge_image_with_mask(
        np.clip(cta_with_smooth_flow[:, :, 256], -0.25, 0.7), vessel_cta_with_smooth_flow[:, :, 256], show=False)
    image = np.concatenate((image_up, image_down), axis=0)
    Functions.image_show(image)

    import visualization.visualize_3d.visualize_stl as stl
    stl.visualize_numpy_as_stl(vessel_cta_with_smooth_flow)


if __name__ == '__main__':
    current_step = 6
    top_dict = '/data_disk/pulmonary_registration/super_precision_register'
    non_hr_ = np.load(top_dict + '/step_0/non_hr.npy')
    cta_hr_ = np.load(top_dict + '/step_' + str(current_step) + '/cta_hr.npy')
    vessel_seg_non_hr_ = np.load(top_dict +
                                 '/step_0/vessel_seg_non_hr.npz')['array']
    vessel_seg_cta_hr_ = np.load(top_dict + '/step_'
                                 + str(current_step) + '/vessel_seg_cta_hr.npz')['array']

    register_and_smooth_flow(non_hr_, cta_hr_, vessel_seg_non_hr_, vessel_seg_cta_hr_)
