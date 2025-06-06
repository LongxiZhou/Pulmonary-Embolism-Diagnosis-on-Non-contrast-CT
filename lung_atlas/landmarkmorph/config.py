import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default='/home/chuy/registration/256*256*256/fixed/xwqg-A00001_2019-01-17.npy')

parser.add_argument("--model_guided", type=str, help="voxelmorph 1 or 2",
                    dest="model_guided", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')

# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="/home/chuy/registration/256*256*256/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=12000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='mse')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=2)
parser.add_argument("--n_save_iter", type=int, help="frequency of model_guided saves",
                    dest="n_save_iter", default=200)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='/home/chuy/registration/256*256*256/model_guided')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')

# test时参数


parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='/home/chuy/Lung_CT')
parser.add_argument("--checkpoint_path", type=str, help="model_guided weight file",
                    dest="checkpoint_path", default="/home/chuy/PycharmProjects/voxelmorph_torch/Checkpoint_model_3/7800.pth")

args = parser.parse_args()


if __name__ == '__main__':
    print(args)