import sys
import training.train_dense_regression as run_model
import torch

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
ibex = False
if not ibex:
    print("not ibex")
    import os

    top_directory = '/home/zhoul0a/Desktop/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU
else:
    top_directory = '/ibex/scratch/projects/c2052/'

TRAIN_DATA_DIR = "/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/training_samples/"
# each sample is an array with path like: TRAIN_DATA_DIR/patient-id_time_slice-id.npy

CHECKPOINT_DIR = "/home/zhoul0a/Desktop/Lung_Altas/Up_sample_Z/stage_one/check_point_refine_L2F64/"
# best model_guided: CHECKPOINT_DIR/direction/fold_saved_model.pth
# current model_guided: CHECKPOINT_DIR/direction/current_fold_saved_model.pth

parameters = {
    "n_epochs": 400,  # this is the maximum epoch, but usually the training finished less than 100 epochs.
    "batch_size": 28,
    "init_features": 64,  # number of kernels for the first convolution layer
    "encoder_blocks": 2,
    "lr": 1e-4,  # learning rate

    #########################################
    # do not forget to change!
    "channels_data": 3,  # the input channels
    "channels_weight": 5,
    "output_channels": 5,  # the output semantics, channel 0 (first channel) is the negative
    "no_rotate": False,  # if the sample shape [x, y], x != y, set it to True
    # the input channel of the sample: window_width * data_channels + enhanced_channels
    # channels: 3 for stage one and 5 for stage two
    #########################################

    'workers': 32,  # num CPU for the parallel data loading
    "train_data_dir": None,
    "test_data_dir": None,  # use train for test
    "checkpoint_dir": None,
    "saved_model_filename": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,  # the fold of training. one in [0, 1, 2, 3, 4] as the program ONLY support five fold.
    "least_loss": None,  # initially the "least_loss" is None
}


def modify_params(test_id=0):
    global TRAIN_DATA_DIR, CHECKPOINT_DIR
    if not TRAIN_DATA_DIR[-1] == '/':
        TRAIN_DATA_DIR = TRAIN_DATA_DIR + '/'
    if not CHECKPOINT_DIR[-1] == '/':
        CHECKPOINT_DIR = CHECKPOINT_DIR + '/'
    parameters["test_id"] = test_id
    parameters["saved_model_filename"] = str(test_id) + "_saved_model.pth"
    train_dict = TRAIN_DATA_DIR
    parameters["train_data_dir"] = train_dict
    parameters["test_data_dir"] = train_dict

    check_point_dict = CHECKPOINT_DIR

    parameters["checkpoint_dir"] = check_point_dict


def training_one_test_id(test_id):
    modify_params(test_id)
    print('test_id', test_id)
    run_model.training(parameters)


def training_all_test_id():
    for test_id in range(5):
        training_one_test_id(test_id)


if __name__ == '__main__':
    """
    the first epoch will be slow if data set exceed GPU Ram, but other epochs will be much faster (if dataset < RAM).
    Set DataLoader(pin_memory=True)
    Make sure batch_size for train and test sum_up * model_size < CUDA memory, otherwise test will compute on CPU.

    The CPU load usually smaller than 5 after the first epoch. if CPU usage is high, maybe model_guided inference is on CPU or
    the dataset exceed the RAM
    """
    training_one_test_id(0)

    exit()
    training_one_test_id(2)
    training_one_test_id(4)
