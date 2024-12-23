
import os
import glob
import warnings

import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data

from lung_atlas.landmarkmorph import losses
from lung_atlas.landmarkmorph.config import args
from lung_atlas.landmarkmorph.datagenerators import Dataset
from lung_atlas.landmarkmorph.model import UNetwork, SpatialTransformer
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
beta = 0


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    ref_img = sitk.GetImageFromArray(ref_img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像
    fixed = np.load(args.atlas_file)
    f_img = fixed[0, :, :, :]
    input_fixed = f_img[np.newaxis, np.newaxis, :]
    vol_size = input_fixed.shape[2:]
    fixed_seg = fixed[2, :, :, :]  # we use the blood vessels as the label points
    print(vol_size)

    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_seg = torch.from_numpy(fixed_seg).to(device).float()

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = UNetwork(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()

    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    dice_loss_fn = losses.dice_loss

    # Get all_file the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.npy'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


    # Training loop.
    for i in range(1, args.n_iter + 1):
        if (i - 1) % args.n_save_iter == 0:
            sum1 = 0
            sum2 = 0
        # Generate the moving images and convert them to tensors.
        input = iter(DL).next()
        #print(input.shape)
        input_moving = input[:, :, :, :, :, 0]
        input_seg = input[:, :, :, :, :, 1]

        # [B, C, D, W, H]
        input_moving = input_moving.to(device).float()
        input_seg = input_seg.to(device).float()
        #print(torch.sum(input_seg * input_seg), torch.sum(input_moving * input_moving))
        #print(input_moving.shape, input_fixed.shape)

        # Run the data through the model_guided to produce warp and flow field
        flow_m2f = UNet(input_moving, input_fixed)
        m2f = STN(input_moving, flow_m2f)
        output_seg = STN(input_seg, flow_m2f)

        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow_m2f)
        #print(torch.sum(input_seg))
        if torch.sum(input_seg) == 0:
            print("0!")
            dice_loss = 0
        else:
            dice_loss = dice_loss_fn(output_seg,  fixed_seg)
        #(beta + 0.3 * i / args.n_save_iter) * dice_loss
        loss = sim_loss + args.alpha * grad_loss + (0.025 + 0.25 * i / args.n_iter) * dice_loss
        sum1 += dice_loss_fn(m2f, input_fixed)
        sum2 += dice_loss
        print("i: %d  loss: %f  sim: %f  grad: %f  dice: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item(), dice_loss), flush=True)
        print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save model_guided checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            print(sum1 / args.n_save_iter)
            print(sum2 / args.n_save_iter)
            # Save images
            #m_name = str(i) + "_m.npy"
            #m2f_name = str(i) + "_m2f.npy"
            #save_image(input_moving, f_img, m_name)
            #save_image(m2f, f_img, m2f_name)
            print("warped images have saved.")
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
