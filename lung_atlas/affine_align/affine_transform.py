"""
see function affine_transform_3d
Inputs: a 3D numpy array in float32, a matrix for transform, a vector for translation
Outputs: affine_transformed 3D numpy array same shape with the input array
"""
import math
import numpy as np
import Tool_Functions.Functions as Functions


# 双线性旋转
def bilinear_rotate(imgArray):
    H, W, channel = imgArray.shape

    pi = math.pi
    theta = 36
    angle = theta * pi / 180

    matrix1 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [-0.5 * H, 0.5 * W, 1]])

    matrix2 = np.array([[math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle), math.cos(angle), 0],
                        [0, 0, 1]])

    matrix3 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0.5 * H, 0.5 * W, 1]])

    new_data = np.zeros_like(imgArray, dtype=np.uint8)

    for i in range(H):
        print(i)
        for j in range(W):

            dot1 = np.matmul(np.array([i, j, 1]), matrix1)
            dot2 = np.matmul(dot1, matrix2)
            dot3 = np.matmul(dot2, matrix3)

            new_coordinate = dot3

            new_i = int(math.floor(new_coordinate[0]))
            new_j = int(math.floor(new_coordinate[1]))

            u = new_coordinate[0] - new_i
            v = new_coordinate[1] - new_j

            if new_j>=W or new_i >=H or new_i<1 or new_j<1 or (i+1)>=H or (j+1)>=W:
                continue

            if (new_i + 1)>=H or (new_j+1)>=W:
                new_data[i, j, :] = imgArray[new_i,new_j, :]

            else:
                new_data[i, j, :] = (1-u)*(1-v)*imgArray[new_i,new_j, :] + \
                                   (1-u)*v*imgArray[new_i,new_j+1, :] + \
                                   u*(1-v)*imgArray[new_i+1,new_j, :] +\
                                   u*v*imgArray[new_i+1,new_j+1, :]

    return new_data


if __name__ == '__main__':
    image = Functions.convert_png_to_np_array('/home/zhoul0a/Desktop/Figure_2.png')[0::2, 0::2, :]
    print(np.shape(image))
    Functions.image_show(image)
    new_image = bilinear_rotate(image)
    Functions.image_show(new_image)
