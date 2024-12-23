import os
import random
import time
import numpy as np
import torch
from classic_models.Unet_3D.utlis import random_flip_rotate_swap, get_labels


def augment_sample(sample, deep_copy=True):
    """

    :param sample: [3, L, L, L], in numpy float16
    :param deep_copy:
    :return: augmented_sample
    """
    if deep_copy:
        sample = np.array(sample, 'float32')

    augment_label = get_labels(swap_axis=True)

    return random_flip_rotate_swap(sample, deep_copy=False, labels=augment_label, reverse=False, show_label=False)


def prepare_tensor(list_sample, device='cuda:0'):
    """

    :param list_sample: [sample, ]
    each sample in shape [3, L, L, L], in numpy float16
    channel 0 is the normalized ct fix (non-contrast) added landmark,
    channel 1 is the normalized ct moving (CTA) added landmark, numpy float16
    channel 2 is the penalty weights for ncc loss based on non-contrast
    :param device:
    :return: fixed_image_tensor, moving_image_tensor, penalty_weight_tensor (torch FloatTensors shaped [N, 1, L, L, L])
    """

    batch_size = len(list_sample)
    image_shape = np.shape(list_sample[0])[1::]

    fixed_image_tensor = np.zeros([batch_size, 1, image_shape[0], image_shape[1], image_shape[2]], 'float32')
    moving_image_tensor = np.zeros([batch_size, 1, image_shape[0], image_shape[1], image_shape[2]], 'float32')
    penalty_weight_tensor = np.zeros([batch_size, 1, image_shape[0], image_shape[1], image_shape[2]], 'float32')

    for i in range(batch_size):
        fixed_image_tensor[i, 0, :, :, :] = list_sample[i][0]
        moving_image_tensor[i, 0, :, :, :] = list_sample[i][1]
        penalty_weight_tensor[i, 0, :, :, :] = list_sample[i][2]

    fixed_image_tensor = torch.FloatTensor(fixed_image_tensor).to(device)
    moving_image_tensor = torch.FloatTensor(moving_image_tensor).to(device)
    penalty_weight_tensor = torch.FloatTensor(penalty_weight_tensor).to(device)

    return fixed_image_tensor, moving_image_tensor, penalty_weight_tensor


def form_batch(list_of_items, augment=True):
    """

    :param augment:
    :param list_of_items: [(sample, whether it is important), ]
    :return: tensors on GPU, list_whether_important
    """
    sample_list = []
    importance_list = []
    for item in list_of_items:
        sample = item[0]
        if augment:
            sample = augment_sample(sample)
        sample_list.append(sample)

        importance_list.append(item[1])

    fixed_image_tensor, moving_image_tensor, penalty_weight_tensor = prepare_tensor(sample_list, device='cuda:0')

    return fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list


# form a list-like object, each item is (sample in numpy float16, sample_importance)
class OriginalSampleDataset:
    """
    Each sample:
    sample is in .npy format,
    in shape [3, L, L, L], here L = 256 or 128 or 64 or 32

    channel 0 is the normalized ct fix (non-contrast) added landmark, in numpy float16 shaped [L, L, L] ,
    mass center of blood vessel set to (L/2, L/2, L/2)

    channel 1 is the normalized ct moving (CTA) added landmark, numpy float16 shaped [L, L, L] ,
    mass center of blood vessel set to (L/2, L/2, L/2)

    channel 2 is the penalty weights for ncc loss based on non-contrast, numpy float16 shaped [L, L, L]


    get item will return: (sample, importance)

    """

    def __init__(self, sample_dir_list=None,  # list of directory for storing CT sample sequences
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 sample_interval=(0, 1),
                 wrong_file_name=None,  # the list of file names to remove
                 important_file_name=None,
                 shuffle_path_list=False
                 ):

        func_load_sample = np.load

        if sample_dir_list is None:
            sample_dir_list = '/data_disk/pulmonary_registration/cast_CTA_to_CT/training_sample_256'
        assert mode in ['train', 'test']
        if type(sample_dir_list) is str:
            sample_dir_list = [sample_dir_list]
        sample_path_list = []
        if important_file_name is None:
            important_file_name = []
        if wrong_file_name is not None:
            wrong_file_name = list(wrong_file_name)
        else:
            wrong_file_name = []

        def process_one_sample_dir(sample_dir):
            name_list_all_samples = os.listdir(sample_dir)
            for name in wrong_file_name:
                if name in name_list_all_samples:
                    print("remove_wrong_file:", name)
                    name_list_all_samples.remove(name)
            for name in name_list_all_samples:
                ord_sum = 0
                for char in name:
                    ord_sum += ord(char)
                if mode == 'train':
                    if ord_sum % 5 == test_id:
                        continue
                    else:
                        sample_path_list.append((os.path.join(sample_dir, name), name in important_file_name))
                else:
                    if ord_sum % 5 == test_id:
                        sample_path_list.append((os.path.join(sample_dir, name), name in important_file_name))
                    else:
                        continue

        for current_sample_dir in sample_dir_list:
            print("getting sample path from:", current_sample_dir)
            process_one_sample_dir(current_sample_dir)

        if shuffle_path_list:
            random.shuffle(sample_path_list)

        self.sample_path_list = sample_path_list[sample_interval[0]:: sample_interval[1]]
        self.length = len(self.sample_path_list)

        print("there are", self.length, "samples")
        print("loading...")
        start_time = time.time()
        self.sample_list = []
        loaded_count = 0
        for idx in range(len(self.sample_path_list)):
            self.sample_list.append((func_load_sample(self.sample_path_list[idx][0]),
                                     self.sample_path_list[idx][1]))
            loaded_count += 1
            if loaded_count % 100 == 0 and loaded_count > 0:
                print(loaded_count, '/', self.length)

        end_time = time.time()
        print("original sample loaded, cost:", end_time - start_time, 's')
        self.pointer = 0
        self.iter_pointer = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample_list[idx % self.length]  # (sample, whether it is important)

    def get_item(self):
        return_value = self.sample_list[self.pointer % self.length]
        self.pointer += 1
        return return_value  # (sample, whether it is important)

    def shuffle(self):
        random.shuffle(self.sample_list)

    def __iter__(self):
        self.iter_pointer = 0
        return self

    def __next__(self):
        if self.iter_pointer >= self.length:
            raise StopIteration()
        item = self.sample_list[self.iter_pointer % self.length]
        self.iter_pointer += 1
        return item


# the dataloader during training or testing
class DataLoaderRegistration:
    """
    Iterative object, prepare data tensors ready for model. Each step return:

    fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list
    """

    def __init__(self, original_sample_dataset, batch_size, shuffle=True, show=True, mode='train', augment=True,
                 drop_last=True):
        """

        :param original_sample_dataset: instance of OriginalSampleDataset
        :param batch_size: batch_size during training or testing
        :param shuffle: shuffle the order of sample
        :param show:
        :param mode: training data is dynamically generated, while the testing data is fixed until
        updating clot simulate parameters
        """
        self.augment = augment
        assert mode in ['train', 'test']
        if show:
            print("mode:", mode)
            print("batch_size:", batch_size, "shuffle:", shuffle, "augment:", augment)

        self.mode = mode
        self.original_sample_dataset = original_sample_dataset
        self.epoch_passed = 0

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_samples = len(original_sample_dataset)
        assert self.num_samples > batch_size

        self.num_batch_processed = 0

    def __len__(self):
        samples_num = self.num_samples
        if samples_num % self.batch_size == 0 or self.drop_last:
            return int(samples_num / self.batch_size)
        return int(samples_num / self.batch_size) + 1

    def __iter__(self):
        print("\n\n#########################################################")
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)
        print("#########################################################")
        if self.shuffle:
            self.original_sample_dataset.shuffle()

        self.num_batch_processed = 0
        return self

    def __next__(self):

        start = self.num_batch_processed * self.batch_size

        if start >= len(self.original_sample_dataset):
            self.epoch_passed += 1
            raise StopIteration()

        end = start + self.batch_size

        if end > len(self.original_sample_dataset):
            if self.drop_last:
                self.epoch_passed += 1
                raise StopIteration()
            else:
                end = len(self.original_sample_dataset)

        item_list = []
        for idx in range(start, end):
            item_list.append(self.original_sample_dataset[idx])

        fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list = form_batch(
            item_list, augment=self.augment)

        self.num_batch_processed += 1
        return fixed_image_tensor, moving_image_tensor, penalty_weight_tensor, importance_list


if __name__ == '__main__':
    temp_sample_dataset = OriginalSampleDataset(
        '/data_disk/pulmonary_registration/cast_CTA_to_CT/training_sample_64', mode='train', test_id=0,
        sample_interval=(0, 1), shuffle_path_list=False)

    temp_dataloader = DataLoaderRegistration(temp_sample_dataset, 4, shuffle=True, show=True,
                                             mode='train', augment=True, drop_last=True)

    def show_tensors(a, b, c):
        import Tool_Functions.Functions as Functions
        length = 64
        image = np.zeros([length, length * 3], 'float32')

        image[:, 0: length] = Functions.cast_to_0_1(c.detach().cpu().numpy()[0, 0, :, :, int(length / 2)])
        image[:, length: int(2 * length)] = Functions.cast_to_0_1(a.detach().cpu().numpy()[0, 0, :, :, int(length / 2)])
        image[:, int(2 * length):] = Functions.cast_to_0_1(b.detach().cpu().numpy()[0, 0, :, :, int(length / 2)])

        Functions.image_show(image)

    batch_count = 0
    for package in temp_dataloader:
        print(batch_count, '/', len(temp_dataloader))
        fix_tensor, move_tensor, weight_tensor, whether_important_list = package
        show_tensors(fix_tensor, move_tensor, weight_tensor)
        print(fix_tensor.shape, move_tensor.shape, weight_tensor.shape, whether_important_list)
        batch_count += 1
    exit()
