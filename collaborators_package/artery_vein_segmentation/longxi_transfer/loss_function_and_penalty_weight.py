import torch
import numpy as np
import analysis.center_line_and_depth_3D as get_center_line_and_depth
import analysis.connect_region_detect as connect_region_detect
import analysis.get_surface_rim_adjacent_mean as get_surface
import pulmonary_embolism.prepare_dataset.get_branch_mask as get_branch_mask
import Tool_Functions.Functions as Functions


def cross_entropy_av_bn_seg(prediction, ground_truth, weight_array, class_balance_weight=(1, 1, 1, 1)):
    """
    
    prediction, ground_truth, weight_tensor in torch.FloatTensor format, on GPU,
    with shape [batch_size, class_num, x, y, z]
    here class_num is 4: they are, artery, vein, blood vessel, and non-blood vessel
    
    :param prediction: [batch_size, class_num, x, y, z], NOT soft_maxed!
    :param ground_truth: [batch_size, class_num, x, y, z], each pixel with value [0, 1]
    :param weight_array: [batch_size, class_num, x, y, z], each pixel with value [0, inf)
    :param class_balance_weight: balance_weight for artery, vein, blood vessel, non-blood vessel
    :return: a float with value [0, inf)
    """

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    softmax = torch.nn.Softmax(dim=1)

    # calculate the loss of blood vessel segmentation
    loss_blood_seg = -softmax_then_log(prediction[:, 2: 4]) * ground_truth[:, 2: 4] * weight_array[:, 2: 4]
    loss_blood_seg[:, 0] = loss_blood_seg[:, 0] * class_balance_weight[2]  # add hyper balance for blood vessel
    loss_blood_seg[:, 1] = loss_blood_seg[:, 1] * class_balance_weight[3]  # add hyper balance for non blood vessel
    loss_blood_seg = torch.sum(loss_blood_seg)
    
    # calculate the loss for artery-vein classification
    loss_av_seg = -softmax_then_log(prediction[:, 0: 2]) * ground_truth[:, 0: 2] * weight_array[:, 0: 2]
    loss_av_seg[:, 0] = loss_av_seg[:, 0] * class_balance_weight[0]  # add hyper balance for artery
    loss_av_seg[:, 1] = loss_av_seg[:, 1] * class_balance_weight[1]  # add hyper balance for vein
    predicted_blood_vessel_probability = softmax(prediction[:, 2: 4])[:, 0]  # probability for the voxel is blood vessel
    loss_av_seg[:, 0] = loss_av_seg[:, 0] * predicted_blood_vessel_probability
    loss_av_seg[:, 1] = loss_av_seg[:, 1] * predicted_blood_vessel_probability
    loss_av_seg = torch.sum(loss_av_seg)
    
    return loss_blood_seg + loss_av_seg


def get_penalty_weight_array_av_bn_seg(rescaled_ct, artery_gt, vein_gt, airway_gt=None, branch_map_artery=None,
                                       branch_map_vein=None, show=True):
    """
    
    penalty weight is the relative penalty for false negative prediction.
    
    all input should has the same shape, in numpy float32
    
    :param show:
    :param rescaled_ct:
    :param artery_gt: 
    :param vein_gt: 
    :param airway_gt: 
    :param branch_map_artery: 
    :param branch_map_vein: 
    :return: penalty weight in shape [4, 512, 512, 512], numpy float32
    """
    
    # refine artery_gt and vein_gt: change them to one simply connected component
    if show:
        print("refine artery_gt and vein_gt: change them to one simply connected component")
    artery_refined = np.zeros(np.shape(artery_gt), 'float32')
    loc_array_artery = Functions.get_location_array(
        connect_region_detect.get_sorted_connected_regions(
            artery_gt, strict=False, show=show)[1])  # get one connected component
    artery_refined[remove_holes(loc_array_artery, add_surface=0, max_hole_radius=1)] = 1  # remove holes

    vein_refined = np.zeros(np.shape(vein_gt), 'float32')
    loc_array_vein = Functions.get_location_array(
        connect_region_detect.get_sorted_connected_regions(
            vein_gt, strict=False, show=show)[1])  # get one connected component
    vein_refined[remove_holes(loc_array_vein, add_surface=0, max_hole_radius=1)] = 1  # remove holes
    
    # remove some known false positives
    if show:
        print("refine artery_gt and vein_gt: remove false positive")
    threshold_fp = Functions.change_to_rescaled(-150)
    potential_vessel_region = np.array(rescaled_ct > threshold_fp, 'float32')
    artery_refined = potential_vessel_region * artery_refined
    vein_refined = potential_vessel_region * vein_refined
    if airway_gt is not None:
        artery_refined = np.clip(artery_refined - airway_gt, 0, 1)
        vein_refined = np.clip(vein_refined - airway_gt, 0, 1)
    
    if branch_map_artery is None:
        if show:
            print("calculate branch_map_artery")
        depth_array_artery = get_center_line_and_depth.get_surface_distance(artery_refined)
        artery_center_line = get_center_line_and_depth.get_center_line(
            artery_refined, surface_distance=depth_array_artery)
        branch_map_artery = get_branch_mask.get_branching_cloud(artery_center_line, depth_array_artery, search_radius=5,
                                                                smooth_radius=1,
                                                                step=1, weight_half_decay=20, refine_radius=4)
        
    if branch_map_vein is None:
        if show:
            print("calculate branch_map_vein")
        depth_array_vein = get_center_line_and_depth.get_surface_distance(vein_refined)
        vein_center_line = get_center_line_and_depth.get_center_line(
            vein_refined, surface_distance=depth_array_vein)
        branch_map_vein = get_branch_mask.get_branching_cloud(vein_center_line, depth_array_vein, search_radius=5,
                                                              smooth_radius=1,
                                                              step=1, weight_half_decay=20, refine_radius=4)
    
    """
    get penalty for artery and vein: 
    sum for root surface is set to 50, the sum for root is set to 100; 
    each root voxel on surface has the same penalty; each inside root voxel has the same penalty 
    penalty for root voxel is 3 times of non-root voxel 
    """
    av_overlap_region = artery_refined * vein_refined  # this part is too ambiguous, set penalty to zero
    if show:
        print("overlap voxel for artery_gt and vein_gt:", np.sum(av_overlap_region))

    artery_effective = artery_refined - av_overlap_region
    surface_artery = get_surface.get_surface(artery_effective, outer=False, strict=False)
    root_artery = np.array(branch_map_artery < 5, 'float32') * artery_effective
    root_artery_surface = surface_artery * root_artery
    non_root_artery = np.array(branch_map_artery >= 5, 'float32') * artery_effective
    non_root_artery_surface = surface_artery * non_root_artery
    refactor_ratio_artery = 50 / np.sum(root_artery)
    refactor_ratio_artery_surface = 50 / np.sum(root_artery_surface)
    if show:
        print("refactor_ratio_artery:", refactor_ratio_artery, 
              "refactor_ratio_artery_surface:", refactor_ratio_artery_surface)
    penalty_artery = root_artery * refactor_ratio_artery + non_root_artery * (refactor_ratio_artery / 3)
    penalty_artery = root_artery_surface * refactor_ratio_artery_surface + non_root_artery_surface * (
            refactor_ratio_artery_surface / 3) + penalty_artery

    vein_effective = vein_refined - av_overlap_region
    surface_vein = get_surface.get_surface(vein_effective, outer=False, strict=False)
    root_vein = np.array(branch_map_vein < 5, 'float32') * vein_effective
    root_vein_surface = surface_vein * root_vein
    non_root_vein = np.array(branch_map_vein >= 5, 'float32') * vein_effective
    non_root_vein_surface = surface_vein * non_root_vein
    refactor_ratio_vein = 50 / np.sum(root_vein)
    refactor_ratio_vein_surface = 50 / np.sum(root_vein_surface)
    if show:
        print("refactor_ratio_vein:", refactor_ratio_vein, 
              "refactor_ratio_vein_surface:", refactor_ratio_vein_surface)
    penalty_vein = root_vein * refactor_ratio_vein + non_root_vein * (refactor_ratio_vein / 3)
    penalty_vein = root_vein_surface * refactor_ratio_vein_surface + non_root_vein_surface * (
            refactor_ratio_vein_surface / 3) + penalty_vein

    """
    get penalty for blood vessel and non-blood vessel:
    the sum of penalty for blood surface is set to 50
    the sum of penalty for blood is set to 100
    the voxel penalty on surface is the same; the voxel penalty inside blood is the same
    the voxel penalty for non-blood vessel is the same
    the sum of penalty for non-blood vessel is 0.1 of the average blood vessel penalty
    """
    blood_vessel_mask = np.clip(artery_refined + vein_refined, 0, 1)
    blood_vessel_surface = get_surface.get_surface(blood_vessel_mask, outer=False, strict=False)
    num_voxel_surface = np.sum(blood_vessel_surface)
    num_voxel_blood = np.sum(blood_vessel_mask)

    penalty_blood = blood_vessel_surface * (50 / num_voxel_surface) + blood_vessel_mask * (50 / num_voxel_blood)
    if show:
        print("penalty weight blood surface:", 50 / num_voxel_surface + 50 / num_voxel_blood)
        print("penalty weight inside blood:", 50 / num_voxel_blood)
    # voxel penalty non_blood is 0.1 of the average blood penalty
    penalty_non_blood = (1 - blood_vessel_mask) * (100 / num_voxel_blood * 0.1)
    if show:
        print("penalty weight non-blood:", 100 / num_voxel_blood * 0.1)

    return np.stack((penalty_artery, penalty_vein, penalty_blood, penalty_non_blood), axis=0)


def remove_holes(loc_list_or_loc_array, add_surface=0, max_hole_radius=2):

    # make sure there is only one connected component !!!

    if type(loc_list_or_loc_array) is list:
        loc_array = Functions.get_location_array(loc_list_or_loc_array)
    else:
        loc_array = loc_list_or_loc_array

    pad = 5 + 2 * add_surface
    assert len(loc_array) == 2 or len(loc_array) == 3

    if len(loc_array) == 3:
        x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
        y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])
        z_min, z_max = np.min(loc_array[2]), np.max(loc_array[2])

        temp_array = np.zeros([int(x_max - x_min + pad), int(y_max - y_min + pad), int(z_max - z_min + pad)], 'float32')

        new_loc_array = (loc_array[0] - x_min + int(pad / 2),
                         loc_array[1] - y_min + int(pad / 2),
                         loc_array[2] - z_min + int(pad / 2))

        temp_array[new_loc_array] = 1
        temp_array = connect_region_detect.convert_to_simply_connected(temp_array, max_hole_radius, add_surface)
        new_loc_array = np.where(temp_array > 0.5)

        loc_array_no_hole = (new_loc_array[0] + x_min - int(pad / 2),
                             new_loc_array[1] + y_min - int(pad / 2),
                             new_loc_array[2] + z_min - int(pad / 2))

        if type(loc_list_or_loc_array) is list:
            return Functions.get_location_list(loc_array_no_hole)
        return loc_array_no_hole
    else:
        x_min, x_max = np.min(loc_array[0]), np.max(loc_array[0])
        y_min, y_max = np.min(loc_array[1]), np.max(loc_array[1])

        temp_array = np.zeros([int(x_max - x_min + pad), int(y_max - y_min + pad)], 'float32')

        new_loc_array = (loc_array[0] - x_min + int(pad / 2),
                         loc_array[1] - y_min + int(pad / 2))

        temp_array[new_loc_array] = 1
        temp_array = connect_region_detect.convert_to_simply_connected(temp_array, max_hole_radius, add_surface)
        new_loc_array = np.where(temp_array > 0.5)

        loc_array_no_hole = (new_loc_array[0] + x_min - int(pad / 2),
                             new_loc_array[1] + y_min - int(pad / 2))

        if type(loc_list_or_loc_array) is list:
            return Functions.get_location_list(loc_array_no_hole)
        return loc_array_no_hole


def check_correctness():
    
    file_name = 'PL00001.npz'
    
    rescaled_ct = np.load('/data_disk/artery_vein_project/new_data/non-contrast/rescaled_ct/' + file_name)['array']
    artery_gt = np.load(
        '/data_disk/artery_vein_project/new_data/non-contrast/ground_truth/artery_gt/' + file_name)['array']
    vein_gt = np.load(
        '/data_disk/artery_vein_project/new_data/non-contrast/ground_truth/vein_gt/' + file_name)['array']
    airway_gt = np.load(
        '/data_disk/artery_vein_project/new_data/non-contrast/ground_truth/airway_gt/' + file_name)['array']

    penalty_weight = get_penalty_weight_array_av_bn_seg(rescaled_ct, artery_gt, vein_gt, airway_gt)

    penalty_artery = penalty_weight[0]
    penalty_vein = penalty_weight[1]
    penalty_blood = penalty_weight[2]
    penalty_non_blood = penalty_weight[3]

    def show_array(input_array):
        for z in range(200, 350, 10):
            Functions.image_show(input_array[:, :, z])

    show_array(penalty_artery)
    show_array(penalty_vein)
    show_array(penalty_blood)
    show_array(penalty_non_blood)


if __name__ == '__main__':
    check_correctness()
