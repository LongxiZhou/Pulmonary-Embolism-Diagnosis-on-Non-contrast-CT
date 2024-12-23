import torch
import torch.nn as nn
import torch.nn.functional as nnf
from classic_models.Unet_3D.U_net_Model_3D import UNet3D, UNet3DSimple, UNet3DSimplest


class RefineRegistrationFlow(nn.Module):
    """
    Given a registration flow, then make it better

    input: fix_image, moving_image, registration_flow
    return: registered_image, refined_registration_flow
    """

    def __init__(self, image_size, num_landmark=16, depth_get_landmark=2, depth_refine_flow=2, inference_phase=True,
                 split_positive_and_negative=False):
        """
        :param image_size: tuple like (256, 256, 256)
        :param num_landmark: the initial feature

        :param depth_get_landmark:
        :param depth_refine_flow:

        depth equals to the num of times get MaxPool

        :param inference_phase: during training and testing, the model will output more tensors for loss
        :param split_positive_and_negative: use independent model for positive and negative flow
        """
        super(RefineRegistrationFlow, self).__init__()
        self.model_dict = {2: UNet3DSimplest, 3: UNet3DSimple, 4: UNet3D}  # key is the depth (number of pooling)
        self.inference_phase = inference_phase
        self.split_positive_and_negative = split_positive_and_negative

        # use this model to extract landmark for moving and fixed
        self.model_extract_landmark = \
            self.model_dict[depth_get_landmark](in_channels=1, out_channels=num_landmark, init_features=num_landmark)

        # landmark from moving and fixed will be concat together
        num_landmark_combined = int(2 * num_landmark)

        self.model_extract_positive_flow = \
            self.model_dict[depth_refine_flow](
                in_channels=num_landmark_combined, out_channels=3, init_features=num_landmark)

        if self.split_positive_and_negative:
            self.model_extract_negative_flow = \
                self.model_dict[depth_refine_flow](
                    in_channels=num_landmark_combined, out_channels=3, init_features=num_landmark)

        self.apply_flow_to_image = RegisterWithGivenFlow(image_size)

        # self.relu_activation = nn.ReLU(inplace=True)

    def apply_flow_to_image_tensor(self, moving_image, registration_flow):
        return self.apply_flow_to_image(moving_image, registration_flow)

    def extract_and_combine_landmark(self, pre_registered_image, fixed_image):
        landmark_pre_registered = self.model_extract_landmark(pre_registered_image)
        landmark_fixed = self.model_extract_landmark(fixed_image)
        landmark_combined = torch.concat((landmark_pre_registered, landmark_fixed), dim=1)
        return landmark_combined  # in shape [N, num_landmark_combined, X, Y, Z]

    def forward(self, moving_image, fixed_image, registration_flow_raw=None):
        """

        :param moving_image: tensor in shape (N, 1, X, Y, Z)
        :param fixed_image: tensor in shape (N, 1, X, Y, Z)
        :param registration_flow_raw: tensor in shape tensor in shape (N, 3, X, Y, Z)
        :return:
        inference_phase:
           return registration_flow_refined, registered_image

        training and testing phase:
           return registration_flow_refined, registered_image
        """
        if registration_flow_raw is not None:
            pre_registered_image = self.apply_flow_to_image_tensor(moving_image, registration_flow_raw)
        else:
            pre_registered_image = moving_image

        landmark_combined = self.extract_and_combine_landmark(pre_registered_image, fixed_image)

        flow_positive = self.model_extract_positive_flow(landmark_combined)

        if self.split_positive_and_negative:
            flow_negative = self.model_extract_negative_flow(landmark_combined)
            final_flow = flow_positive - flow_negative
        else:
            final_flow = flow_positive

        if registration_flow_raw is not None:
            final_flow = registration_flow_raw + final_flow

        if self.inference_phase:
            return final_flow, self.apply_flow_to_image_tensor(moving_image, final_flow)
        else:
            return final_flow, self.apply_flow_to_image_tensor(moving_image, final_flow)


class RegisterWithGivenFlow(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """

        :param size: tuple like (256, 256, 256)
        :param mode:
        """
        assert len(size) == 3 or len(size) == 2
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # [D, X, Y, Z]
        grid = torch.unsqueeze(grid, 0)  # [1, D, X, Y, Z]
        # grid = grid.type(torch.FloatTensor)
        grid = grid.float()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, moving_image, flow_move_to_register):
        """

        :param moving_image: tensor in shape (N, C, X, Y, Z), or (N, C, X, Y)
        :param flow_move_to_register: tensor in shape (N, 3, X, Y, Z), or (N, 2, X, Y)
        :return: registered tensor same shape with input:

            registered[0, :, x + flow[0, 0, x, y, z], y + flow[0, 1, x, y, z], z + flow[0, 2, x, y, z]] =
            moving_image[0, :, x, y, z]

        """
        # locations for src, relative to registered
        old_locs = self.grid - flow_move_to_register  # (N, 3, X, Y, Z)
        shape = flow_move_to_register.shape[2:]  # (X, Y, Z)

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            old_locs[:, i, ...] = 2 * (old_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            old_locs = old_locs.permute(0, 2, 3, 1)  # (N, X, Y, 2)
            # the channels need to be reversed, as this is locations for src, relative to registered
            old_locs = old_locs[..., [1, 0]]
        elif len(shape) == 3:
            old_locs = old_locs.permute(0, 2, 3, 4, 1)  # (N, X, Y, Z, 3)
            # the channels need to be reversed, as this is locations for src, relative to registered
            old_locs = old_locs[..., [2, 1, 0]]

        return nnf.grid_sample(moving_image, old_locs, align_corners=True, mode=self.mode)


def linear_rescale_registration_flow(flow_tensor, new_shape):
    """

    :param flow_tensor: (B, C, X_old, Y_old, Z_old)
    :param new_shape: (X_new, Y_new, Z_new)
    :return: tensor in shape (B, C, X_new, Y_new, Z_new)
    """
    from torch.nn.functional import interpolate
    return interpolate(flow_tensor, new_shape, mode='trilinear', align_corners=True)


if __name__ == '__main__':
    exit()
