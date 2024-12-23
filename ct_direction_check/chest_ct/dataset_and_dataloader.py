import Tool_Functions.Functions as Functions
import os
import time
import random
import torch
import numpy as np


def process_to_tensor(batch_list):
    """
    tensors on CPU

    :param batch_list:
    :return: float_tensor in shape [batch_size, 1, 768, 256], float_tensor in shape [batch_size, 48]
    """
    batch_size = len(batch_list)

    # on CPU, numpy float32 array and float tensor share memory
    ground_truth_array = np.zeros([batch_size, 48], 'float32')
    input_array = np.zeros([batch_size, 1, 768, 256], 'float32')

    for i in range(len(batch_list)):
        sample = batch_list[i]
        image, class_id = sample
        input_array[i] = image
        ground_truth_array[i, class_id] = 1

    return torch.FloatTensor(input_array), torch.FloatTensor(ground_truth_array)


# form a list-like object
class SampleDataset:
    """
    Each sample:
    (numpy_array float16 in [3, 256, 256], class_id)
    """

    def __init__(self, sample_dir_list=None,  # directory or list of directory for storing CT sample sequences
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=(0, 5),  # use ord_sum % test_id[1] == test_id[0] as the test samples
                 sample_interval=(0, 1),
                 wrong_file_name=None,  # the list of file names to remove
                 batch_size=32,
                 drop_last=True
                 ):
        assert sample_dir_list is not None
        assert mode in ['train', 'test']
        if type(sample_dir_list) is str:
            sample_dir_list = [sample_dir_list]
        scan_path_list = []
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
                assert name[-7::] == '.pickle'
                ord_sum = 0
                for char in name:
                    ord_sum += ord(char)

                if mode == 'train':
                    if ord_sum % test_id[1] == test_id[0]:
                        continue
                    else:
                        scan_path_list.append(os.path.join(sample_dir, name))
                else:
                    if ord_sum % test_id[1] == test_id[0]:
                        scan_path_list.append(os.path.join(sample_dir, name), )
                    else:
                        continue

        for current_sample_dir in sample_dir_list:
            print("getting sample path from:", current_sample_dir)
            process_one_sample_dir(current_sample_dir)

        self.scan_path_list = scan_path_list[sample_interval[0]:: sample_interval[1]]

        print("there are", len(self.scan_path_list), "scans under:", sample_dir_list)
        print("loading...")
        start_time = time.time()
        self.sample_list = []

        loaded_count = 0
        for idx in range(len(self.scan_path_list)):
            sample_list_48 = Functions.pickle_load_object(self.scan_path_list[idx])
            self.sample_list = self.sample_list + sample_list_48
            loaded_count += 1
            if loaded_count % 1000 == 0 and loaded_count > 0:
                print(loaded_count, '/', len(self.scan_path_list))

        self.length = len(self.sample_list)

        end_time = time.time()
        print("original sample loaded, cost:", end_time - start_time, 's')
        print("loaded sample:", len(self.sample_list))

        self.pointer = 0
        self.iter_pointer = 0
        self.batch_passed = 0

        self.iteration_passed = 0
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample_list[idx % self.length]  # (sample, class_id)

    def get_item(self):
        return_value = self.sample_list[self.pointer % self.length]
        self.pointer += 1
        return return_value

    def random_get_item(self):
        return self.sample_list[random.randint(0, self.length - 1)]

    def get_batch(self, batch_size, random_select=False):
        sample_list = []
        for i in range(batch_size):
            if not random_select:
                sample = self.get_item()
            else:
                sample = self.random_get_item()
            sample_list.append(sample)
        return sample_list

    def shuffle(self):
        random.shuffle(self.sample_list)

    def __iter__(self):
        self.iter_pointer = 0
        self.batch_passed = 0
        print("###############################")
        print("#  iteration passed:", self.iteration_passed)
        print("###############################")
        return self

    def __next__(self):

        start_id = self.iter_pointer

        if start_id >= self.length:
            self.iteration_passed += 1
            raise StopIteration()

        end_id = self.iter_pointer + self.batch_size
        if end_id >= self.length:
            if self.drop_last:
                end_id = self.length
            else:
                self.iteration_passed += 1
                raise StopIteration()

        batch_list = self.sample_list[start_id: end_id]

        input_tensor, ground_truth_tensor = process_to_tensor(batch_list)

        self.iter_pointer = end_id

        self.batch_passed += 1

        return input_tensor, ground_truth_tensor


if __name__ == '__main__':
    sample_dataset = SampleDataset(sample_dir_list='/data_disk/chest_ct_direction/training_samples/not_clip',
                                   sample_interval=(0, 100), batch_size=32, mode='test')

    print(len(sample_dataset))

    sample_dataset.shuffle()

    for input_, ground_truth_ in sample_dataset:
        print(input_.shape, ground_truth_.shape, sample_dataset.batch_passed)

    exit()

    for i_ in range(10):
        image_, class_id_ = sample_dataset[i_]
        print(np.shape(image_), class_id_)
