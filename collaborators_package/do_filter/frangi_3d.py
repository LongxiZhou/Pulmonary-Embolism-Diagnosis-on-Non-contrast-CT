import cv2
from skimage.filters import frangi
import numpy as np
from analysis.get_surface_rim_adjacent_mean import get_surface


def window_transform(img, win_min, win_max):
    for i in range(img.shape[0]):
        img[i] = 255.0*(img[i] - win_min)/(win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255)/img[i].max()
        img[i] = img[i]*c
    return img.astype(np.uint8)


def sigmoid(img, alpha, beta):
    return 1 / (1 + np.exp((beta - img) / alpha))


def vessel_enhancement(image, label):
    image = image * 1600 - 600
    wintrans = window_transform(image, -1000.0, 650.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    label = cv2.erode(label, kernel)
    roi = wintrans * label
    roi_sigmoid = sigmoid(roi, 30, 60)
    roi_frangi = frangi(roi_sigmoid, sigmas=range(1, 5, 1), alpha=0.25, beta=0.25, gamma=75, black_ridges=False)
    cv2.normalize(roi_frangi, roi_frangi, 0, 1, cv2.NORM_MINMAX)
    thresh = np.percentile(sorted(roi_frangi[roi_frangi > 0]), 95)
    vessel = (roi_frangi - thresh) * (roi_frangi > thresh) / (1 - thresh)
    vessel[vessel > 0] = 1
    vessel[vessel <= 0] = 0
    vessel = vessel * (label - get_surface(label))
    vessel = vessel * np.array(image > 0, "float32")
    return vessel
