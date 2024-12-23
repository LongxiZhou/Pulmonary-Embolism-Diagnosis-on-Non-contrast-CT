import argparse, os
import torch
import math, random
from scipy.ndimage import zoom
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from visualization.visualize_3d import visualize_stl as view
from semantic_segmentation.artery_vein.utils import TrainSetLoader
from semantic_segmentation.artery_vein.model import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=5e-6, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=15)
parser.add_argument('--gamma', type=float, default=1, help='Learning Rate decay')


def dice_loss(array_1, array_2):
    inter = torch.sum(array_1 * array_2)
    norm = torch.sum(array_1 * array_1) + torch.sum(array_2 * array_2)
    return 1 - 2 * (inter + 0.1) / (norm + 0.1)


def overlap_loss(array_1, array_2):
    inter = torch.sum(array_1[:, 0] * array_2[:, 1]) + torch.sum(array_1[:, 1] * array_2[:, 0])
    return inter / (torch.sum(array_1 * array_1) + torch.sum(array_2 * array_2))


def train():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    opt.seed = random.randint(1, 10000)
    # opt.seed = 1468
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    print("===> Building model")
    model = UNet(in_channel=1, num_classes=3)
    model.load_state_dict(torch.load(
            "/data/Train_and_Test/segmentation/av_model_celoss/model_epoch_130.pth"))

    print("===> Setting GPU")
    model = model.cuda()
    model = model.to('cuda')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        train_set = TrainSetLoader('/data/Train_and_Test/segmentation/artery_vein_CTA', device)
        training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

        trainor(training_data_loader, optimizer, model, epoch, scheduler)
        scheduler.step()


def trainor(training_data_loader, optimizer, model, epoch, scheduler):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    ce_loss_obj = nn.BCEWithLogitsLoss()
    for iteration, (raw, img) in enumerate(training_data_loader):
        # print(raw.shape, img.shape)
        pre = model(raw)
        # pre = pre.detach().cpu().numpy()
        # pre = zoom(pre, 2)
        # view.visualize_two_numpy(pre[0, 1], pre[0, 0])

        ce_loss = ce_loss_obj(pre, img)
        loss = dice_loss(pre[:, 0], img[:, 0]) + dice_loss(pre[:, 1], img[:, 1])
        loss += ce_loss
        loss += overlap_loss(pre, img)
        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration + 1)))

    save_checkpoint(model, epoch)


def save_checkpoint(model, epoch):
    model_out_path = "/data/Train_and_Test/segmentation/av_model_celoss/" \
                     + "model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


train()


