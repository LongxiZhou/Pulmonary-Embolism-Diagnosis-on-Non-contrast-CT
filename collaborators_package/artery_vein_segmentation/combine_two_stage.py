from analysis import connectivity_refine_fast as connectivity
import numpy as np


def combine_together(label_1, label_2, connect_num=3):
    artery_1 = np.array(label_1[0] > 0.6, "float32")  # You can adjust the threshold by yourself
    vein_1 = np.array(label_1[1] > 0.6, "float32")

    artery_2 = np.array(label_2[0] > 0.6, "float32")
    vein_2 = np.array(label_2[1] > 0.6, "float32")

    artery = artery_1 + artery_2 - artery_1 * artery_2
    vein = vein_1 + vein_2 - vein_1 * vein_2

    artery = connectivity.select_region(artery, connect_num)
    vein = connectivity.select_region(vein, connect_num)

    return artery, vein




