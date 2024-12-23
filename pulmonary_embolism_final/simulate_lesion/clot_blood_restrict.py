"""
some clot shaped like clot, like deep venous thrombosis
"""
import Tool_Functions.Functions as Functions
import numpy as np
from smooth_mask.get_lung_vessel_blood_region.dataset_and_dataloader import SimulateLesionDataset
import os
import random
import analysis.point_cloud as point_cloud
import analysis.connect_region_detect as connect_region_detect

top_dict_blood_vessel = '/data_disk/RAD-ChestCT_dataset/semantics/blood_mask/'
top_dict_center_line = '/data_disk/RAD-ChestCT_dataset/depth_and_center-line/blood_center_line/'
fn_list = os.listdir(top_dict_blood_vessel)
save_path_clot = '/data_disk/artery_vein_project/extract_blood_region/' \
                 'lesion_simulation/list-of-loc-array_DVT_volume>250_one_connected_0.pickle'

existing_lesion_dataset = SimulateLesionDataset(difficulty_level=8, iterative_loading=True)

if os.path.exists(save_path_clot):
    list_of_loc_array = Functions.pickle_load_object(save_path_clot)
else:
    list_of_loc_array = []


def get_qualified_lesions(num_qualified=5):
    file_name = fn_list[random.randint(0, len(fn_list) - 1)]

    center_line = np.load(top_dict_center_line + file_name)['array']
    vessel_mask = np.load(top_dict_blood_vessel + file_name)['array']

    loc_list_center_line = Functions.get_location_list(np.where(center_line > 0.5))

    failed_count = 0

    qualified_loc_array_list = []
    while len(qualified_loc_array_list) < num_qualified:
        center_line_loc = loc_list_center_line[random.randint(0, len(loc_list_center_line) - 1)]
        lesion_loc_array = existing_lesion_dataset.get_arbitrary_lesion()
        while lesion_loc_array is None:
            lesion_loc_array = existing_lesion_dataset.get_arbitrary_lesion()
        lesion_loc_array_potential = get_potential_loc_array(vessel_mask, lesion_loc_array, center_line_loc)
        print("volume:", len(lesion_loc_array_potential[0]))
        if len(lesion_loc_array_potential[0]) >= 250:
            qualified_loc_array_list.append(lesion_loc_array_potential)
        else:
            failed_count += 1
        if failed_count > 50:
            return get_qualified_lesions(num_qualified=num_qualified)
    return qualified_loc_array_list


def get_potential_loc_array(vessel_mask, lesion_loc_array, center_line_loc):
    lesion_loc_array_new = point_cloud.set_mass_center(lesion_loc_array, center_line_loc,
                                                       int_loc_arrays=True, show=False)

    vessel_mask[lesion_loc_array_new] += vessel_mask[lesion_loc_array_new]
    lesion_loc_array_potential = np.where(vessel_mask > 1.5)
    vessel_mask[lesion_loc_array_new] -= vessel_mask[lesion_loc_array_new]

    if len(lesion_loc_array_potential[0]) == 0:
        return [], [], []

    bounding_array = point_cloud.point_cloud_to_numpy_array(lesion_loc_array_potential, pad=2)
    bounding_array_refine = connect_region_detect.refine_connected_component(bounding_array, number_leave=1)

    lesion_loc_array_potential = np.where(bounding_array_refine > 0.5)
    lesion_loc_array_potential = point_cloud.set_mass_center(lesion_loc_array_potential, (0, 0, 0), show=False,
                                                             int_loc_arrays=True)

    return lesion_loc_array_potential


def generate_lesion_simulation(save_interval=1000, terminate=10000):
    loc_array_list = list_of_loc_array

    while len(loc_array_list) < terminate:
        print("current qualified lesions:", len(loc_array_list))
        if len(loc_array_list) > 0 and len(loc_array_list) % save_interval == 0:
            print("saving...")
            Functions.pickle_save_object(save_path_clot, loc_array_list)
        loc_array_list = loc_array_list + get_qualified_lesions(5)


if __name__ == '__main__':
    generate_lesion_simulation()
