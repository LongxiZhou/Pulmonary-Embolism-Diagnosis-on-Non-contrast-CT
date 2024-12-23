import numpy as np
from analysis.center_line_and_depth_3D import get_center_line
from analysis.connectivity_yuetan import select_region
import analysis.get_surface_rim_adjacent_mean as get_surface


def refinement(raw, artery_root, vein_root, blood, twice_refinement=False, iteration=1, max_parallel_count=24):
    for i in range(iteration):
        artery_root += get_surface.get_surface(artery_root, outer=True, strict=False)
        vein_root += get_surface.get_surface(vein_root, outer=True, strict=False)

    vein_potential = np.clip(blood - artery_root, 0, 1)
    vein_mask_new = select_region(vein_potential, 4)
    artery_mask_new = np.clip(blood - vein_mask_new, 0, 1)

    if twice_refinement:
        valid_mask = np.array(raw > 0.2, "float32")
        artery_mask_new *= valid_mask
        vein_mask_new *= valid_mask

        artery_axis = get_center_line(artery_mask_new, max_parallel_count=max_parallel_count)
        vein_axis = get_center_line(vein_mask_new, max_parallel_count=max_parallel_count)

        artery_mask_new -= get_surface.get_surface(artery_mask_new, outer=False)
        artery_mask_new += get_surface.get_surface(artery_mask_new, outer=True)

        vein_mask_new -= get_surface.get_surface(vein_mask_new, outer=False)
        vein_mask_new += get_surface.get_surface(vein_mask_new, outer=True)

        artery_axis = artery_axis + get_surface.get_surface(artery_axis, outer=True, strict=True)
        vein_axis = vein_axis + get_surface.get_surface(vein_axis, outer=True, strict=True)

        artery_mask_new += artery_axis
        vein_mask_new += vein_axis

    return np.array(artery_mask_new > 0, 'float32'), np.array(vein_mask_new > 0, 'float32')
