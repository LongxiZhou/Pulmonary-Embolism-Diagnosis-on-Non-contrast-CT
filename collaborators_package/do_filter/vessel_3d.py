from scipy import ndimage
import numpy as np


class Vessel_Filter3d:
    def __init__(self, image, sigma, tau):
        super(Vessel_Filter3d, self).__init__()

        self.image = image
        self.sigma = sigma
        self.tau = tau
        self.size = image.shape

    def gradient_3d(self, np_array, option):
        x_size, y_size, z_size = self.size
        gradient = np.zeros(np_array.shape)
        if option == "x":
            gradient[0] = np_array[1] - np_array[0]
            gradient[x_size - 1] = np_array[x_size - 1] - np_array[x_size - 2]
            gradient[1:x_size - 2] = \
                (np_array[2:x_size - 1] - np_array[0:x_size - 3]) / 2
        elif option == "y":
            gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
            gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
            gradient[:, 1:y_size - 2] = \
                (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
        else:
            gradient[:, :, 0] = np_array[:, :, 1] - np_array[:, :, 0]
            gradient[:, :, z_size - 1] = np_array[:, :, z_size - 1] - np_array[:, :, z_size - 2]
            gradient[:, :, 1:z_size - 2] = \
                (np_array[:, :, 2:z_size - 1] - np_array[:, :, 0:z_size - 3]) / 2
        return gradient

    def Hessian_3d(self, image, sigma):
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        Dz = self.gradient_3d(image, "z")
        Dzz = self.gradient_3d(Dz, "z")

        Dy = self.gradient_3d(image, "y")
        Dyy = self.gradient_3d(Dy, "y")

        Dx = self.gradient_3d(image, "x")
        Dxx = self.gradient_3d(Dx, "x")

        Dxy = self.gradient_3d(Dx, 'y')
        Dyz = self.gradient_3d(Dy, 'z')
        Dzx = self.gradient_3d(Dz, 'x')
        return np.array([Dxx, Dyy, Dzz, Dxy, Dyz, Dzx])

    def eigval_Hessian3d(self, Dxx, Dyy, Dxy):
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

    def imageEigenvalues(self, volume, sigma):
        # print(self.Hessian_3d(volume, sigma))
        H = self.Hessian_3d(volume, sigma) * (sigma ** 2)

        # # reduce computation by computing vesselness only where needed
        B1 = -(H[0] + H[1] + H[2])
        B2 = H[0] * H[1] + H[1] * H[2] + H[2] * H[0] - (H[3] ** 2 + H[4] ** 2 + H[5] ** 2)
        B3 = H[0] * H[4] ** 2 + H[1] * H[5] ** 2 + H[2] * H[3] ** 2 -\
             (H[0] * H[1] * H[2] + 2 * H[3] * H[4] * H[5])

        T = np.ones(B1.shape)
        T[(B1 <= 0)] = 0
        T[(B2 <= 0) & (B3 == 0)] = 0
        T[(B1 > 0) & (B2 > 0) & (B1 * B2 < B3)] = 0

        index = np.where(T == 1)[0]
        for i in range(6):
            H[i] = H[i][index]

        [lambda1i, lambda2i, lambda3i] = self.eigvalOfhessian2d(H)
        lambda1 = np.zeros(T.shape)
        lambda2 = np.zeros(T.shape)
        lambda3 = np.zeros(T.shape)

        lambda1[index] = lambda1i
        lambda2[index] = lambda2i
        lambda3[index] = lambda3i

        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0
        lambda3[(np.isinf(lambda3))] = 0

        return lambda1, lambda2, lambda3

    def vessel_3d(self):
        vessel_ness = None
        for j in range(len(self.sigma)):
            _, lambda2, lambda3 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda2 = lambda2 * (-1)
            lambda3 = lambda3 * (-1)

            lambda_rho = lambda3.copy()
            new_tau = self.tau * np.max(lambda3)
            lambda_rho[(lambda3 > 0) & (lambda3 <= new_tau)] = new_tau
            lambda_rho[lambda3 <= 0] = 0

            different = lambda_rho - lambda2
            response = ((lambda2 ** 2) * different) * 27 / \
                       ((lambda2 + lambda_rho) ** 3)
            response[(lambda2 >= lambda_rho / 2) & (lambda_rho > 0)] = 1
            response[(lambda2 <= 0) or (lambda_rho <= 0)] = 0
            response[np.where(np.isinf(response))[0]] = 0

            if j == 0:
                vessel_ness = response
            else:
                vessel_ness = np.maximum(vessel_ness, response)

        return vessel_ness


