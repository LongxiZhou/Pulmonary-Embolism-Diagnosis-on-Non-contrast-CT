import numpy as np
import scipy.ndimage as ndimage
import Tool_Functions.Functions as Functions
from analysis.connect_region_detect import get_sorted_connected_regions

img = np.load('/data_disk/CTA-CT_paired-dataset/dataset_non_contrast/PE_High_Quality/depth_and_center-line/depth_array/11.17p04.npz')['array']
img = np.array(img, 'float32')
print(np.shape(img))

max_depth = np.max(img)

loc_array_max = np.where(img == max_depth)

x_max = int(np.average(loc_array_max[0]))
y_max = int(np.average(loc_array_max[1]))
z_max = int(np.average(loc_array_max[2]))

# Get local maximum values of desired neighborhood
# I'll be looking in a 5x5x5 area
img2 = ndimage.maximum_filter(img, size=(5, 5, 5))

img[np.where(img == 0)] = -1

candidate_loc_array = np.where(img == img2)
print(len(candidate_loc_array[0]))
new_array = img2 * 0
new_array[candidate_loc_array] = 1


sorted_loc_dict = get_sorted_connected_regions(new_array, strict=False, show=True)

for key, value in sorted_loc_dict.items():
    loc_array = Functions.get_location_array(value)
    if np.mean(img[loc_array]) > 15:
        print(key, np.mean(img[loc_array]), max_depth)

exit()

Functions.image_show(new_array[x_max, :, :])
Functions.image_show(new_array[:, y_max, :])
Functions.image_show(new_array[:, :, z_max])

exit()

# Threshold the image to find locations of interest
# I'm assuming 6 standard deviations above the mean for the threshold
img_thresh = img2.mean() + img2.std() * 3

# Since we're looking for maxima find areas greater than img_thresh

labels, num_labels = ndimage.label(img2 > img_thresh)

# Get the positions of the maxima
coords = ndimage.measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))

# Get the maximum value in the labels
values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))

print(coords)
print(values)