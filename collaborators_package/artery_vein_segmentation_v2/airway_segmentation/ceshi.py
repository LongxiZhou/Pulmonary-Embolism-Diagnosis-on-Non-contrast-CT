import numpy as np
import nibabel as nib
from visualization.visualize_3d import visualize_stl as view
from scipy.ndimage import binary_fill_holes
import cv2
import matplotlib.pyplot as plt


def regionGrowing(grayImg, seed, threshold):
    """
    :param grayImg: 灰度图像
    :param seed: 生长起始点的位置
    :param threshold: 阈值
    :return: 取值为{0, 255}的二值图像
    """
    [maxX, maxY, maxZ] = grayImg.shape[0:3]

    # 用于保存生长点的队列
    pointQueue = []
    pointQueue.append((seed[0], seed[1], seed[2]))
    outImg = np.zeros_like(grayImg)
    outImg[seed[0], seed[1], seed[2]] = 1

    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1], seed[2]])

    # 用于计算生长点周围26个点的位置
    Next26 = [[-1, -1, -1], [-1, 0, -1], [-1, 1, -1],
              [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
              [-1, 0, 1], [-1, 0, 0], [-1, 0, -1],
              [0, -1, -1], [0, 0, -1], [0, 1, -1],
              [0, 1, 0], [-1, 0, -1],
              [0, -1, 0], [0, -1, 1], [-1, 0, -1],
              [0, 0, 1], [1, 1, 1], [1, 1, -1],
              [1, 1, 0], [1, 0, 1], [1, 0, -1],
              [1, -1, 0], [1, 0, 0], [1, -1, -1]]

    while (len(pointQueue) > 0):
        # 取出队首并删除
        growSeed = pointQueue[0]
        del pointQueue[0]

        for differ in Next26:
            growPointx = growSeed[0] + differ[0]
            growPointy = growSeed[1] + differ[1]
            growPointz = growSeed[2] + differ[2]

            if ((growPointx < 0) or (growPointx > maxX - 1) or
                    (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)):
                continue

            if (outImg[growPointx, growPointy, growPointz] == 1):
                continue

            data = grayImg[growPointx, growPointy, growPointz]
            if abs(data - pointsMean) < threshold:
                pointsNum += 1
                pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                outImg[growPointx, growPointy, growPointz] = 1
                pointQueue.append([growPointx, growPointy, growPointz])

    return outImg


def FillHole(mask):
    mask = np.array(mask, np.uint8)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # len_contour = len(contours)
    # print(len_contour)
    # contour_list = []
    # for i in range(len_contour):
    #     drawing = np.zeros_like(mask, np.uint8)  # create a black image
    #     img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
    #     contour_list.append(img_contour)
    # out = sum(contour_list)
    drawing = mask.copy()  # create a black image
    drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)
    out = cv2.drawContours(drawing, contours, -1, (255, 255, 255), -1)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

    return out


np_array = np.load("/data/chest_CT/rescaled_ct/non-contrast/rescaled_ct/PL00042.npz")["array"]
# plt.imshow(np_array[:, :, 255], cmap="gray")
# plt.show()
np_array = np_array * 1600 - 600
seed = (180, 240, 255)  # 选择合适的种子点，在这里选择气管的位置
threshold = 80  # 临近像素的阈值，微调可以减少误分割，和少分割
airway = regionGrowing(np_array, seed, threshold)
view.visualize_numpy_as_stl(airway)
