import numpy as np
import analysis.connect_region_detect as connected_region
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.visualize_stl as stl
from pulmonary_embolism_v2.simulate_lesion.simulate_lesion_surface_growth import remove_holes
from analysis.get_surface_rim_adjacent_mean import get_surface


def decrease_diameter(rescaled_ct, depth_of_raw_vessel_mask, blood_branch_map, blood_center_line,
                      remove_hole=True, show=True):

    valid_region = np.array(rescaled_ct > Functions.change_to_rescaled(-200))

    refined_vessel_mask = np.array(depth_of_raw_vessel_mask > 0.5, 'float32')

    refined_vessel_mask = connected_region.refine_connected_component(
            refined_vessel_mask, 2, None, strict=False, show=show, leave_min=0.25)

    if remove_hole:
        loc_array = np.where(refined_vessel_mask > 0.5)
        loc_array = remove_holes(loc_array)
        refined_vessel_mask[loc_array] = 1

    loc_list_blood_region_rescaled = Functions.get_location_list(np.where(depth_of_raw_vessel_mask > 0.5))
    for loc in loc_list_blood_region_rescaled:
        if depth_of_raw_vessel_mask[loc] < (7 - blood_branch_map[loc]):
            refined_vessel_mask[loc] = 0

    blood_center_line = blood_center_line + get_surface(blood_center_line, outer=True, strict=False)

    refined_vessel_mask = np.clip(refined_vessel_mask + blood_center_line, 0, 1) * valid_region

    refined_vessel_mask = connected_region.refine_connected_component(
        refined_vessel_mask, 2, None, strict=False, show=show, leave_min=0.25)

    return refined_vessel_mask


def decrease_diameter_v2(rescaled_ct, center_line_vessel, extend_time=5, strict=False, show=True):
    valid_region = np.array(rescaled_ct > Functions.change_to_rescaled(-200))
    center_line_vessel = connected_region.refine_connected_component(
        center_line_vessel, 2, None, strict=False, show=show, leave_min=0.25)
    for i in range(extend_time):
        center_line_vessel = center_line_vessel + get_surface(center_line_vessel, outer=True, strict=strict)
        center_line_vessel = center_line_vessel * valid_region

    return center_line_vessel


if __name__ == '__main__':
    patient = '11.17p01.npz'
    rescaled_ct_ = \
    np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/rescaled_ct-denoise/' + patient)['array']
    vessel_mask_ = \
    np.load('/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/semantics/blood_mask_high_recall/' + patient)[
        'array']
    blood_center_line_ = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/depth_and_center-line/blood_high_recall_center_line/' + patient)[
        'array']

    print(np.sum(vessel_mask_))
    vessel_depth_ = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/depth_and_center-line/high_recall_depth_array/' + patient)[
        'array']
    print(np.sum(vessel_depth_ > 0))

    blood_branch_map_ = np.load(
        '/data_disk/CTA-CT_paired-dataset/dataset_CTA/PE_High_Quality/depth_and_center-line/high_recall_blood_branch_map/' + patient)[
        'array']
    print(np.sum(blood_branch_map_ > 0))

    mask = decrease_diameter(rescaled_ct_, vessel_depth_, blood_branch_map_, blood_center_line_)
    stl.visualize_numpy_as_stl(mask)
    exit()

