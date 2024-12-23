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
from registration_pulmonary.models.register import VxmDense as Register_1
from torch.nn.functional import interpolate
import torch.nn.functional as F
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=4000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000)
parser.add_argument("--resume", default="/home/chuy/Artery_Vein_Upsampling/checkpoint/pretrained_discriminator",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=2, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=1, help='Learning Rate decay')


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

    # print("The step_by_step register is")
    # step_scale = [192, 256, 384, 512]
    # print(step_scale)

    # print("===> Building model_0")
    # scale_factor = 4
    # vol_size = [384, 512, 320]
    # nf_enc = [32, 64, 128, 256]
    # nf_dec = [256, 128, 64, 64, 64, 32, 32]
    # model_0 = Register_0(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    # print("===> Loading datasets")
    #
    # model_0.load_state_dict(
    #     torch.load("/home/chuy/registration/checkpoint/96/model_epoch_172.pth"))
    # model_0 = model_0.cuda()
    # model_0 = model_0.to('cuda:0')

    print("===> Building model_1")
    scale_factor = 2
    vol_size = [256, 256, 256]
    nf_enc = [16, 32, 64, 128]
    nf_dec = [128, 128, 64, 64, 32, 32, 16]
    model_1 = Register_1(inshape=vol_size, unet_encoder=nf_enc, unet_decoder=nf_dec, scale=scale_factor)
    print("===> Loading datasets")
    model_1.load_state_dict(
        torch.load("/home/chuy/Train_and_Test/registration/lung_register/128_model/model_epoch_320.pth"))

    # model = model.cuda()
    # model = model.to('cuda:0')

    print("===> Loading datasets")
    train_set = TrainSetLoader("/home/chuy/Train_and_Test/registration/lung_register/512", device)
    training_data_loader_1 = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                        shuffle=True)
    training_data_loader_2 = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                        shuffle=True)

    print("===> Setting GPU")
    # model_0 = model_0.cuda()
    # model_0 = model_0.to('cuda:0')

    model_1 = model_1.cuda()
    model_1 = model_1.to('cuda:0')
    # refine_model = refine_model.cuda()
    # refine_model = refine_model.to('cuda:0')

    num_params = sum(param.numel() for param in model_1.parameters())
    print(num_params)
    print(model_1)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model_1.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(300, opt.nEpochs + 1):
        print(epoch)
        trainor(optimizer, model_1, training_data_loader_1, training_data_loader_2, epoch, scheduler)


def trainor(optimizer, model_1, loader_1, loader_2, epoch, scheduler):
    # print("===> Loading datasets")
    # train_set = TrainSetLoader_Random(device)
    scheduler.step()
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model_1.train()
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
        # registered_img_0, registered_seg_0, _ = model_0(moving_img, fixed_img, moving_seg, fixed_seg)

        (registered_img_1, registered_seg_1, pos_flow) = \
            model_1(moving_img, fixed_img, moving_seg, fixed_seg)

        dice_loss_1 = dice_loss_fn(registered_img_1, fixed_img)
        dice_loss_2 = dice_loss_fn(registered_seg_1, fixed_seg)
        # mean_loss = mae_loss_fn(registered_img_1, fixed_img)
        # mask_mean_loss = mae_loss_fn(registered_img_1 * registered_seg_1, fixed_img * fixed_seg)
        grad_loss = grad_loss_fn(pos_flow)

        loss = dice_loss_1 + grad_loss + dice_loss_2

        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("vessel {:.5f} airway: {:.5f} lung: {:.5f}".format(l_vessel, l_airway, l_lung))
        print("===> Epoch[{}]: DiceLoss_1: {:.5f} DiceLoss_2: {:.5f} Grad_Loss: {:.5f} loss_avg: {:.5f}".format
              (epoch, dice_loss_1, dice_loss_2, grad_loss, loss_epoch / (iteration % 100 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model_1, epoch, "/home/chuy/Train_and_Test/registration/lung_register/128_model/")
            print("model has benn saved")
    save_checkpoint(model_1, epoch, "/home/chuy/Train_and_Test/registration/lung_register/128_model/")
    # save_checkpoint(refine_model, epoch, "/home/chuy/Artery_Vein_Upsampling/checkpoint/pretrained_seg/refine/")
    print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = path + "model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))


train()