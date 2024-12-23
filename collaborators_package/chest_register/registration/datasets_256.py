import os
from scipy.ndimage import zoom
import numpy as np
from denoising.predict_denoised import predict_denoised
from analysis.connectivity import select_region
# path = "/home/chuy/registration/512*512*512/f031.npz"
# file = np.load(path)["arr_0"]
# print(file.shape)
# exit()
# artery = np.load("/data/chest_CT/semantic_seg/normal_xwzc/blood_mask/xwzc000015_unknown.npz")["array"]
# view.visualize_numpy_as_stl(artery)
# exit()
img_path = "/data/chest_CT/rescaled_ct/xwzc"
blood_path = "/data/chest_CT/semantic_seg/normal_xwzc/blood_mask"
save_path = "/home/chuy/Train_and_Test/registration/lung_register/512"
file_list = np.sort(os.listdir(img_path))
for filename in file_list:
    print(filename)
    img = np.load(os.path.join(img_path, filename))["arr_0"]
    img = predict_denoised(img)
    blood = np.load(os.path.join(blood_path, filename.replace("npy", "npz")))["array"]
    blood = select_region(blood, num=1)

    file = np.zeros([2, 256, 256, 256])
    file[0] = zoom(img, 1 / 2)
    file[1] = np.array(zoom(blood, 1 / 2) >= 0.5, "float32")

    np.save(os.path.join(save_path, filename.replace("npz", "npy")), file)
