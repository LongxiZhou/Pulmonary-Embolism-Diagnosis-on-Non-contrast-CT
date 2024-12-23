import numpy as np
import os
import torch
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


def get_labels(swap_axis=True):

    label_flip = (np.random.rand() > 0.5, np.random.rand() > 0.5, np.random.rand() > 0.5)
    label_rotate = (np.random.randint(4), np.random.randint(4), np.random.randint(4))
    if swap_axis:
        label_swap = np.random.randint(6)
    else:
        label_swap = 0

    return label_flip, label_rotate, label_swap


def random_flip(sample, deep_copy=True, label_flip=None, reverse=False):
    """

    :param reverse:
    flipped = random_flip(original, label_flip=label reverse=False)
    original = random_flip(flipped, label_flip=label reverse=True)
    :param label_flip:
    :param deep_copy:
    :param sample: numpy float32
    :return:
    """

    if reverse:
        assert label_flip is not None

    def flip_on_axis(ts, label):
        if not reverse:
            if label[0]:
                ts = np.flip(ts, (0,))
            if label[1]:
                ts = np.flip(ts, (1,))
            if label[2]:
                ts = np.flip(ts, (2,))
        else:
            if label[2]:
                ts = np.flip(ts, (2,))
            if label[1]:
                ts = np.flip(ts, (1,))
            if label[0]:
                ts = np.flip(ts, (0,))
        return ts

    if deep_copy:
        sample = np.array(sample, 'float32')
    shape = np.shape(sample)
    if label_flip is None:
        label_flip = (np.random.rand() > 0.5, np.random.rand() > 0.5, np.random.rand() > 0.5)
    assert 3 <= len(shape) <= 4
    if len(shape) == 3:
        sample = np.reshape(sample, [1, shape[0], shape[1], shape[2]])

    channel = np.shape(sample)[0]
    for i in range(channel):
        sample[i] = flip_on_axis(sample[i], label_flip)

    if len(shape) == 3:
        return sample[0]
    return sample


class RandomRotateWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        label = (np.random.randint(4), np.random.randint(4), np.random.randint(4))
        transformed = [RandomRotateWithWeight.rotate_on_axis(sample[k], label) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def rotate_on_axis(ts, label):
        ts = torch.rot90(ts, label[0], (1, 2))
        ts = torch.rot90(ts, label[1], (2, 3))
        ts = torch.rot90(ts, label[2], (3, 1))
        return ts


def random_rotate(sample, deep_copy=True, label_rotate=None, reverse=False):
    """

    :param reverse:
    :param label_rotate:
    :param deep_copy:
    :param sample: numpy float32
    :return:
    """
    if reverse:
        assert label_rotate is not None
        label_rotate = ((4 - label_rotate[0]) % 4, (4 - label_rotate[1]) % 4, (4 - label_rotate[2]) % 4)

    def rotate_on_axis(ts, label):
        if not reverse:
            if not label[0] == 0:
                ts = np.rot90(ts, label[0], (0, 1))
            if not label[1] == 0:
                ts = np.rot90(ts, label[1], (1, 2))
            if not label[2] == 0:
                ts = np.rot90(ts, label[2], (2, 0))
        else:
            if not label[2] == 0:
                ts = np.rot90(ts, label[2], (2, 0))
            if not label[1] == 0:
                ts = np.rot90(ts, label[1], (1, 2))
            if not label[0] == 0:
                ts = np.rot90(ts, label[0], (0, 1))
        return ts

    if deep_copy:
        sample = np.array(sample, 'float32')
    shape = np.shape(sample)
    if label_rotate is None:
        label_rotate = (np.random.randint(4), np.random.randint(4), np.random.randint(4))
    assert 3 <= len(shape) <= 4
    if len(shape) == 3:
        sample = np.reshape(sample, [1, shape[0], shape[1], shape[2]])

    channel = np.shape(sample)[0]
    for i in range(channel):
        sample[i] = rotate_on_axis(sample[i], label_rotate)

    if len(shape) == 3:
        return sample[0]
    return sample


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


def random_swap_axis(sample, deep_copy=True, label_swap=None, reverse=False):
    """

    :param reverse:
    :param label_swap:
    :param deep_copy:
    :param sample: numpy float32
    :return:
    """

    def swap_axis(ts, label):

        if reverse:
            if label == 3:
                ts = np.transpose(ts, (1, 2, 0))
            if label == 4:
                ts = np.transpose(ts, (2, 0, 1))

        if label == 0:
            return ts  # (0, 1, 2)
        if label == 1:
            return np.transpose(ts, (0, 2, 1))  # (0, 2, 1)
        if label == 2:
            return np.transpose(ts, (1, 0, 2))  # (1, 0, 2)
        if label == 3:
            return np.transpose(ts, (1, 2, 0))  # (1, 2, 0)
        if label == 4:
            return np.transpose(ts, (2, 0, 1))  # (2, 0, 1)

        return np.transpose(ts, (2, 1, 0))  # (2, 1, 0)

    if deep_copy:
        sample = np.array(sample, 'float32')
    shape = np.shape(sample)
    if label_swap is None:
        label_swap = np.random.randint(6)
    assert 3 <= len(shape) <= 4
    if len(shape) == 3:
        sample = np.reshape(sample, [1, shape[0], shape[1], shape[2]])

    channel = np.shape(sample)[0]
    for i in range(channel):
        sample[i] = swap_axis(sample[i], label_swap)

    if len(shape) == 3:
        return sample[0]
    return sample


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


def random_flip_rotate_swap(sample, deep_copy=True, labels=None, reverse=False, swap_axis=True, show_label=False):
    """

    :param show_label:
    :param swap_axis: may swap axis
    :param reverse: True for flip_rotate_swap, False for swap_rotate_flip
    :param labels: label_flip, label_rotate, label_swap
    :param sample: numpy float32, shape [channel, x, y, z] or [x, y, z]

    all channels will undergone same flip, rotate, swap

    :param deep_copy:
    :return:
    """
    if deep_copy:
        sample = np.array(sample, 'float32')
    if labels is None:
        labels = get_labels(swap_axis)
    if show_label:
        print("label_flip, label_rotate, label_swap", labels)
    label_flip, label_rotate, label_swap = labels
    if not reverse:
        sample = random_flip(sample, deep_copy=False, label_flip=label_flip)
        sample = random_rotate(sample, deep_copy=False, label_rotate=label_rotate)
        sample = random_swap_axis(sample, deep_copy=False, label_swap=label_swap)
    else:
        sample = random_swap_axis(sample, deep_copy=False, label_swap=label_swap, reverse=True)
        sample = random_rotate(sample, deep_copy=False, label_rotate=label_rotate, reverse=True)
        sample = random_flip(sample, deep_copy=False, label_flip=label_flip, reverse=True)

    return sample


if __name__ == '__main__':

    """
    rotate & swap, swap & flip, rotate & flip   all form complete 48 augment versions
    """

    random_array = np.random.rand(7, 7, 7)

    unique_list = [random_array, ]
    label_for_unique = [((False, False, False), (0, 0, 0), 0)]  # label_flip, label_rotate, label_swap

    for i_ in range(1000):
        if i_ % 100 == 0:
            print(i_)

        # new_array = random_flip_rotate_swap(random_array)

        label_flip_ = (np.random.rand() > 0.5, np.random.rand() > 0.5, np.random.rand() > 0.5)
        new_array = random_flip(random_array, label_flip=label_flip_)
        label_swap_ = np.random.randint(6)
        new_array = random_swap_axis(new_array, label_swap=label_swap_)

        unseen = True
        for value in unique_list:
            if np.average(np.abs(value - new_array)) < 1e-7:
                unseen = False
                break
        if unseen:
            unique_list.append(np.array(new_array))
            label_for_unique.append((label_flip_, (0, 0, 0), label_swap_))
            print("new version", len(unique_list), 'detected at', i_)

    print(len(unique_list))
    print(label_for_unique)
    print(len(label_for_unique))
    print(len(set(label_for_unique)))

    set_label = set(label_for_unique)
    for label in label_for_unique:
        count = 0
        for item in label_for_unique:
            if hash(label) == hash(item):
                count += 1
        if count > 1:
            print(label)

    exit()
    import Tool_Functions.Functions as Functions
    Functions.pickle_save_object('/home/zhoul0a/Desktop/Longxi_Platform/format_convert/label_augment.pickle',
                                 label_for_unique)
    exit()
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
