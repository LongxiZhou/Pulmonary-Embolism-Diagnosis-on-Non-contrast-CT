from scipy import ndimage
import numpy as np
from analysis.get_surface_rim_adjacent_mean import get_surface
from analysis.connectivity_yuetan import select_region


class vesselness2d:
    def __init__(self, image, sigma, tau):
        super(vesselness2d, self).__init__()

        self.image = image
        self.sigma = sigma
        self.tau = tau
        self.size = image.shape

    def gaussian_filter(self, image, sigma):
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        return image

    def gradient_2d(self, np_array, option):
        x_size = self.size[0]
        y_size = self.size[1]
        gradient = np.zeros(np_array.shape)
        if option == "x":
            gradient[0, :] = np_array[1, :] - np_array[0, :]
            gradient[x_size - 1, :] = np_array[x_size - 1, :] - np_array[x_size - 2, :]
            gradient[1:x_size - 2, :] = \
                (np_array[2:x_size - 1, :] - np_array[0:x_size - 3, :]) / 2
        else:
            gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
            gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
            gradient[:, 1:y_size - 2] = \
                (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
        return gradient

    def Hessian2d(self, image, sigma):
        # print(sigma)
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        Dy = self.gradient_2d(image, "y")
        Dyy = self.gradient_2d(Dy, "y")

        Dx = self.gradient_2d(image, "x")
        Dxx = self.gradient_2d(Dx, "x")
        Dxy = self.gradient_2d(Dx, 'y')
        return Dxx, Dyy, Dxy

    def eigval_Hessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        # hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        hyy = -c * hyy
        hxy = -c * hxy

        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]
        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()
        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]
        #     lambda1i, lambda2i = hessian_matrix_eigvals([hxx, hyy, hxy])
        lambda1i, lambda2i = self.eigval_Hessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # removing noise
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2

    def vesselness2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                        (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        #     vesselness = vesselness / np.max(vesselness)
        vesselness[(vesselness < 1e-2)] = 0
        #         vesselness = vesselness.reshape(self.size)
        return vesselness


def vessel_enhance_xyz(raw_array, lung=None, reverse=False):
    sigma = [0.5, 1, 1.5]
    tau = 0.75
    raw_array = np.clip((raw_array * 1600 + 400) / 1400, 0, 1)
    # raw_array = 1 / (1 + np.exp((80 - raw_array) / 20))
    raw_array *= lung

    if reverse:
        raw_array = raw_array * 255
    else:
        raw_array = 255 - raw_array * 255

    prediction_1 = np.zeros(raw_array.shape)
    prediction_2 = np.zeros(raw_array.shape)
    prediction_3 = np.zeros(raw_array.shape)

    for j in range(raw_array.shape[2]):

        raw_1 = raw_array[:, :, j]
        if np.sum(raw_1) == 0:
            continue
        output_1 = vesselness2d(raw_1, sigma, tau)
        output_1 = output_1.vesselness2d()
        prediction_1[:, :, j] = output_1

    for j in range(raw_array.shape[1]):
        raw_2 = raw_array[:, j]
        if np.sum(raw_2) == 0:
            continue
        output_2 = vesselness2d(raw_2, sigma, tau)
        output_2 = output_2.vesselness2d()
        prediction_2[:, j] = output_2

    for j in range(raw_array.shape[0]):
        raw_3 = raw_array[j]
        if np.sum(raw_3) == 0:
            continue
        output_3 = vesselness2d(raw_3, sigma, tau)
        output_3 = output_3.vesselness2d()
        prediction_3[j] = output_3

    prediction = np.array(prediction_1 * prediction_2 * prediction_3 > 0.7, "float32")

    sub_lung = lung - get_surface(lung, strict=False)
    sub_lung = sub_lung - get_surface(sub_lung, strict=False)
    prediction = prediction * sub_lung

    prediction = select_region(prediction, 2)
    return prediction


def vessel_enhance(raw_array, lung=None):
    sigma = [0.5, 1, 1.5]
    tau = 0.75

    raw_array = 255 - raw_array * 255

    prediction = np.zeros(raw_array.shape)

    for j in range(raw_array.shape[-1]):

        raw_1 = raw_array[:, :, j]
        if np.sum(raw_1) == 0:
            continue
        output_1 = vesselness2d(raw_1, sigma, tau)
        output_1 = output_1.vesselness2d()
        prediction[:, :, j] = output_1

    prediction = np.array(prediction > 0.8, "float32")
    sub_lung = lung - get_surface(lung, strict=False)
    sub_lung = sub_lung - get_surface(sub_lung, strict=False)
    prediction = prediction * sub_lung
    # prediction = select_region(prediction, 2)
    return prediction


def slice_filter(raw_array, tau=1):
    sigma = [0.5, 1, 1.5]

    raw_array = 255 - np.clip(raw_array + 0.25, 0, 1) * 255
    output = vesselness2d(raw_array, sigma, tau)
    output = output.vesselness2d()
    return output

