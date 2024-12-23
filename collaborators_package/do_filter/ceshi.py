import numpy as np
from analysis.get_surface_rim_adjacent_mean import get_surface
from analysis.connectivity_yuetan import select_region
from visualization.visualize_3d import visualize_stl as view
from filter.vessel_2d import vesselness2d

sigma = [0.5, 1]
tau = 0.5
# img = np.load("/data/chest_CT/rescaled_ct/f0xx/TM-7.npy")[:, :, 255]
# plt.imshow(img, cmap="gray")
# plt.show()
#
# img = 1 - np.clip(img + 0.25, 0, 1)
#
# output = vesselness2d(img, sigma, tau)
# output = output.vesselness2d()
#
# plt.imshow(output, cmap="gray")
# plt.show()
# exit()
raw_array = np.load("/data/chest_CT/rescaled_ct/non-contrast/rescaled_ct/PL00001.npz")["array"]
raw_array = 255 - np.clip(raw_array + 0.25, 0, 1) * 255
lung = np.load("/data/chest_CT/rescaled_ct/non-contrast/semantics/lung_mask/PL00001.npz")["array"]
# prediction = jerman.jerman_vessel_3d(raw_array, sigma, tau)
# # plt.imshow(prediction[:, :, 255], cmap="gray")
# # plt.show()
# # print(np.max(prediction), np.min(prediction))
# prediction = np.array(prediction > 0.1, "float32")
# view.visualize_numpy_as_stl(prediction * lung)
# exit()
# #
# # prediction = prediction * (lung - get_surface(lung, outer=False))
# # prediction = select_region(prediction, 2)
# # view.visualize_numpy_as_stl(prediction)

prediction_1 = np.zeros([512, 512, 512])
prediction_2 = np.zeros([512, 512, 512])
prediction_3 = np.zeros([512, 512, 512])
# loc = np.array(np.where(lung > 0))
# z_max, z_min = np.max(loc[:, -1]) + 30, np.min(loc[:, -1]) - 30
# print(z_min, z_max)
for j in range(512):

    raw_1 = raw_array[:, :, j]
    output_1 = vesselness2d(raw_1, sigma, tau)
    output_1 = output_1.vesselness2d()
    prediction_1[:, :, j] = output_1

    raw_2 = raw_array[:, j]
    output_2 = vesselness2d(raw_2, sigma, tau)
    output_2 = output_2.vesselness2d()
    prediction_2[:, j] = output_2

    raw_3 = raw_array[j]
    output_3 = vesselness2d(raw_3, sigma, tau)
    output_3 = output_3.vesselness2d()
    prediction_3[j] = output_3

    # print(np.sum(prediction))
# # view.visualize_numpy_as_stl(prediction)
prediction = np.array(prediction_1 * prediction_2 * prediction_3> 0.7, "float32")
prediction = prediction * (lung - get_surface(lung, outer=False))
prediction = select_region(prediction, 2)
view.visualize_numpy_as_stl(prediction)

