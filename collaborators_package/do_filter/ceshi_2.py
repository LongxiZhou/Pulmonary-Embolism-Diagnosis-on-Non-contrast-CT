import numpy as np
from visualization.visualize_3d import visualize_stl as view
import matplotlib.pyplot as plt
from filter.frangi import frangi

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
raw_array = np.clip(raw_array + 0.25, 0, 1) * 255
lung = np.load("/data/chest_CT/rescaled_ct/non-contrast/semantics/lung_mask/PL00001.npz")["array"]
prediction = np.array(frangi.frangi(raw_array) > 0.7, "float32")

plt.imshow(prediction[:, :, 255], cmap="gray")
plt.show()

view.visualize_numpy_as_stl(prediction)