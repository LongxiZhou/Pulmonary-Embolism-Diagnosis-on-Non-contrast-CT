import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from registration_pulmonary.utils import TrainSetLoader
from registration_pulmonary.models.register import VxmDense as Register_0
from torch.nn.functional import interpolate
from registration_pulmonary.models.losses import ncc_loss
import torch.nn.functional as F
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=4000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000)
parser.add_argument("--resume", default="/home/chuy/Artery_Vein_Upsampling/checkpoint/pretrained_discriminator",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.5, help='Learning Rate decay')


def mae_loss_fn(input, target):
    return torch.mean(torch.abs(input - target))


def dice_loss_fn(input, target):
    intersect = torch.sum(input * target)
    denominator = torch.sum(input * input) + torch.sum(target * target)
    return 1 - 2 * (intersect / denominator)


def grad_loss_fn(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    return grad


def train():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("The step_by_step register is")
    step_scale = [192, 256, 384, 512]
    print(step_scale)

    print("===> Building model")
    scale_factor = 4
    vol_size = [512, 512, 512]
    nf_enc = [32, 64, 128, 256]
    nf_dec = [256, 128, 64, 64, 64, 32, 32]
    model = Register_0(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    print("===> Loading datasets")
    train_set = TrainSetLoader("/home/chuy/registration/lung_register_without_heart", device)
    training_data_loader_1 = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                        shuffle=True)
    training_data_loader_2 = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                        shuffle=True)

    # model.load_state_dict(
    #     torch.load("/home/chuy/registration/checkpoint/192/model_epoch_1.pth"))
    # refine_model = UNetRefine(in_channels=7, out_channels=5)
    # model.load_state_dict(
    #     torch.load("/home/chuy/Artery_Vein_Upsampling/checkpoint/pretrained_seg/u2net_5/backup/model_epoch_3.pth"))

    print("===> Setting GPU")
    model = model.cuda()
    model = model.to('cuda:0')
    # refine_model = refine_model.cuda()
    # refine_model = refine_model.to('cuda:0')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        trainor(optimizer, model, training_data_loader_1, training_data_loader_2, epoch, scheduler)


def trainor(optimizer, model, loader_1, loader_2, epoch, scheduler):
    # print("===> Loading datasets")
    # train_set = TrainSetLoader_Random(device)
    scheduler.step()
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    # loss_seg_dice = DiceLoss4MOTS(num_classes=2).to(device)
    # loss_seg_ce = CELoss4MOTS(num_classes=2, ignore_index=255).to(device)

    for iteration, (fixed, moving) in enumerate(zip(loader_1, loader_2)):
        moving_img = moving[0]
        moving_seg = moving[1]
        # print(moving.shape, moving_seg.shape)
        # mode = "trilinear"
        # moving_img = interpolate(moving_img, scale_factor=scale, mode=mode)
        # moving_seg = interpolate(moving_seg, scale_factor=scale, mode=mode)
        # moving_seg[moving_seg > 0.1] = 1

        fixed_img = fixed[0]
        fixed_seg = fixed[1]
        # print(moving_img.shape, moving_seg.shape, fixed_seg.shape, fixed_img.shape)
        # fixed_img = interpolate(fixed_img, scale_factor=scale, mode=mode)
        # fixed_seg = interpolate(fixed_seg, scale_factor=scale, mode=mode)
        # fixed_seg[fixed_seg > 0.1] = 1

        (registered_img, registered_seg, pos_flow) = model(moving_img, fixed_img, moving_seg, fixed_seg)

        # dice_loss = ncc_loss(registered_seg, fixed_seg)
        mean_loss = ncc_loss(registered_img, fixed_img)
        mask_mean_loss = ncc_loss(registered_img * registered_seg, fixed_img * fixed_seg)
        grad_loss = grad_loss_fn(pos_flow)

        loss = mean_loss + 2 * mask_mean_loss + grad_loss * 4

        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("vessel {:.5f} airway: {:.5f} lung: {:.5f}".format(l_vessel, l_airway, l_lung))
        print("===> Epoch[{}]: Loss: {:.5f} Loss: {:.5f} Loss: {:.5f} loss_avg: {:.5f}".format
              (epoch, dice_loss, mean_loss, mask_mean_loss, loss_epoch / (iteration % 100 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/home/chuy/registration/512_model/")
            print("model has benn saved")
    save_checkpoint(model, epoch, "/home/chuy/registration/512_model/")
    # save_checkpoint(refine_model, epoch, "/home/chuy/Artery_Vein_Upsampling/checkpoint/pretrained_seg/refine/")
    print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = path + "model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))


train()
