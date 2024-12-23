import numpy as np
import os
import torch
import torchvision.transforms
import glob


class RandomFlipWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        label = (np.random.rand() > 0.5, np.random.rand() > 0.5, np.random.rand() > 0.5)
        transformed = [RandomFlipWithWeight.flip_on_axis(sample[k], label) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def flip_on_axis(ts, label):
        if label[0]:
            ts = torch.flip(ts, (1,))
        if label[1]:
            ts = torch.flip(ts, (2,))
        if label[2]:
            ts = torch.flip(ts, (3,))
        return ts


class RandomRotateWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        label = (np.random.randint(4), np.random.randint(4), np.random.randint(4))
        transformed = [RandomRotateWithWeight.rotate_on_axis(sample[k], label) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def rotate_on_axis(ts, label):
        torch.rot90(ts, label[0], (1, 2))
        torch.rot90(ts, label[1], (2, 3))
        torch.rot90(ts, label[2], (3, 1))
        return ts


class SwapAxisWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        label = np.random.randint(6)
        transformed = [SwapAxisWithWeight.swap_axis(sample[k], label) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def swap_axis(ts, label):
        if label == 0:
            return ts  # (1, 2, 3)
        if label == 1:
            return torch.transpose(ts, 2, 3)  # (1, 3, 2)
        if label == 2:
            return torch.transpose(ts, 1, 2)  # (2, 1, 3)
        if label == 3:
            ts = torch.transpose(ts, 1, 3)
            return torch.transpose(ts, 1, 2)  # (2, 3, 1)
        if label == 4:
            ts = torch.transpose(ts, 1, 2)
            return torch.transpose(ts, 1, 3)  # (3, 1, 2)

        return torch.transpose(ts, 1, 3)  # (3, 2, 1)


class ToTensorWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        transformed = [torch.from_numpy(sample[k])
                       for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))


class WeightedTissueDataset3D(torch.utils.data.Dataset):
    """
    sample should be in shape [channels_data + channels_weight + channels_gt, :, :, :]
    """
    def __init__(self, sample_dir,
                 image_pattern_or_list="*.npy",
                 transform=None,
                 channels_data=3,
                 channels_weight=2,
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use patient_id % 5 == test_id as the test patients
                 ):
        self.sample_dir = sample_dir
        sample_list = None
        if type(image_pattern_or_list) == str:
            sample_list = [os.path.basename(f) for f in glob.glob(os.path.join(sample_dir, image_pattern_or_list))]
        elif type(image_pattern_or_list) == list:
            sample_list = image_pattern_or_list
        assert sample_list is not None

        if channels_weight == 0:
            print("feature enhanced weight is defaulted!!!")
        else:
            print("we have", channels_weight, "channels for the penalty weights")
        print("# all_file sample files:", len(sample_list))
        if mode == 'train':
            sample_files_filtered = [fn for fn in sample_list if not int(fn.split('_')[0][-1]) % 5 == test_id]
            print("# number of training samples:", len(sample_files_filtered))
        else:
            sample_files_filtered = [fn for fn in sample_list if int(fn.split('_')[0][-1]) % 5 == test_id]
            print("# number of testing samples:", len(sample_files_filtered))
        self.sample_files = np.array(sample_files_filtered).astype(np.string_)

        self.transform = transform
        self.channels_data = channels_data
        self.channels_weight = channels_weight
        self.length = len(self.sample_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length
        sample_array = np.load(os.path.join(self.sample_dir, self.sample_files[idx].decode('utf-8')))
        image = sample_array[:self.channels_data, :, :, :]
        label = sample_array[(self.channels_data + self.channels_weight):, :, :, :]

        if self.channels_weight == 0:
            weight = np.ones(np.shape(label), 'float32')
        else:
            weight = sample_array[self.channels_data: (self.channels_data + self.channels_weight), :, :, :]

        sample = {"image": image, "label": label, 'weight': weight}
        if self.transform:
            return self.transform(sample)
        else:
            return sample


if __name__ == '__main__':
    """
    composed_transform = torchvision.transforms.Compose([
        ToTensorWithWeight(),
        RandomFlipWithWeight(),
        RandomRotateWithWeight(),
        SwapAxisWithWeight()
    ])
    import Tool_Functions.Functions as Functions
    sample = np.load('/home/zhoul0a/Desktop/pulmonary nodules/data_v2/training_samples_3d/1_2020-05-01_2.npy')
    print(np.shape(sample))
    test_sample = {"image": sample[:3, :, :, :], "label": sample[5:, :, :, :], 'weight': sample[3: 5, :, :, :]}
    test_sample = composed_transform(test_sample)
    for i in range(3):
        Functions.image_show(test_sample["image"][i, 32, :, :])
    for i in range(2):
        Functions.image_show(test_sample["weight"][i, 32, :, :])
    for i in range(2):
        Functions.image_show(test_sample["label"][i, 32, :, :])
    """
    exit()
