import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
from visualization.visualize_3d import visualize_stl as view
from denoising.model_2.random_pattern_generator import pattern_generator
from semantic_segmentation.airway_segmentation.loss import focal_loss, dice_loss, sad_loss
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from semantic_segmentation.airway_segmentation.get_model import get_model
from semantic_segmentation.airway_segmentation.utils import TrainSetLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.85
                    , help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    cuda = opt.cuda
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = get_model(cubesize=[80, 192, 304])

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    checkpoint = torch.load('/data/Train_and_Test/segmentation/airway_model/baseline_fr_ad.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(checkpoint)
    model = model.cuda()
    model = model.to('cuda')

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        raw_set = TrainSetLoader('/data/Train_and_Test/segmentation/airway', device)
        raw_loader = DataLoader(dataset=raw_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
        trainor(raw_loader, optimizer, model, epoch)
        scheduler.step()
        # seg_scheduler.step()


def trainor(raw_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    for iteration, (raw, reference, coord) in enumerate(raw_loader):
        # print(torch.max(reference), torch.mean(reference))
        # if torch.sum(reference[0]) == 0:
        #     continue
        predict, attentions = model(raw, coord)
        # predict_ex = predict.cpu().detach().numpy()[0, 0]
        # view.visualize_numpy_as_stl(predict_ex)
        # print(np.sum(predict_ex))
        # refer_ex = reference.cpu().detach().numpy()[0, 0]
        # view.visualize_two_numpy(predict_ex, refer_ex)
        # exit()
        # predict[predict < 0.2] = 0
        # loss_1 = focal_loss(predict, reference)

        # predict_bi = predict
        # predict_bi[predict_bi > 0.5] = 1
        # predict_bi[predict_bi < 0.5] = 0
        loss_2 = dice_loss(predict, reference)
        # case_pred = predict[0]
        # ds6, ds7, ds8 = predict[1], predict[2], predict[3]
        # loss_2 = dice_loss(case_pred, reference) + \
        #          dice_loss(ds6, reference) + \
        #          dice_loss(ds7, reference) + \
        #          dice_loss(ds8, reference)
        # print(loss_1, loss_2)
        loss = loss_2

        gamma_sad = [0.1, 0.1, 0.1]
        for iter_sad in range(2):
            loss += (gamma_sad[iter_sad]) * sad_loss(attentions[iter_sad], attentions[iter_sad + 1],
                                                     encoder_flag=True)

        gamma_sad = [0.1, 0.1, 0.1]
        for iter_sad in range(3, 6):
            loss += (gamma_sad[iter_sad - 3]) * sad_loss(attentions[iter_sad], attentions[iter_sad + 1],
                                                         encoder_flag=False)

        loss.backward()
        optimizer.step()
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 200 + 1)))

        if (iteration + 1) % 200 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/data/Train_and_Test/segmentation/airway_model/")
            # save_checkpoint(seg_model, epoch, "/home/chuy/Artery_Vein_Upsampling/checkpoint/whole/segment/")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = path + "model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))


train()


