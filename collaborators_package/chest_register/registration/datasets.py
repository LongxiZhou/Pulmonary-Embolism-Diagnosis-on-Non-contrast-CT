import os
from scipy.ndimage import zoom
import numpy as np
from analysis.connectivity import select_region
# path = "/home/chuy/registration/512*512*512/f031.npz"
# file = np.load(path)["arr_0"]
# print(file.shape)
# exit()
artery = np.load("/data/chest_CT/semantic_seg/normal_xwzc/blood_mask/xwzc000015_unknown.npz")["array"]
# view.visualize_numpy_as_stl(artery)
# exit()
img_path = "/data/chest_CT/rescaled_ct/xwzc"
blood_path = "/data/chest_CT/semantic_seg/normal_xwzc/blood_mask"
save_path = "/home/chuy/Train_and_Test/registration/lung_register/128"
file_list = np.sort(os.listdir(img_path))
for filename in file_list:
    print(filename)
    img = np.load(os.path.join(img_path, filename))["arr_0"]
    blood = np.load(os.path.join(blood_path, filename.replace("npy", "npz")))["array"]
    blood = select_region(blood, num=1)

    file = np.zeros([2, 128, 128, 128])
    file[0] = zoom(img, 1 / 4)
    file[1] = np.array(zoom(blood, 1 / 4) >= 0.25, "float32")

    np.save(os.path.join(save_path, filename.replace("npz", "npy")), file)
