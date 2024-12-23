import os
import random
import Tool_Functions.Functions as Functions
from pulmonary_embolism_final.simulate_lesion.simulate_clot import \
    random_select_clot_sample_dict, apply_clot_on_sample
from pulmonary_embolism_final.utlis.sample_to_tensor_simulate_clot import prepare_tensors_simulate_clot
from pulmonary_embolism_final.utlis.sample_to_tensor_with_annotation import prepare_tensors_with_annotation
from pulmonary_embolism_final.utlis.data_augmentation import random_flip_rotate_swap_sample, get_labels
from functools import partial
from Tool_Functions.file_operations import extract_all_file_path
import multiprocessing
import time
import numpy as np
import torch


# form dataset for clot samples
class ClotDataset:
    """
    Each item is a clot_sample_dict:

    clot_sample_dict: a dict, with key "loc_clot_set", "loc_depth_set" and "range_clot"
    clot_sample_dict["loc_clot_set"] = {(x, y, z), }
    clot_sample_dict["clot_depth_dict"] = {(x, y, z): b, ..., 'max_depth': max_depth}  here b is the clot depth
    the mass center for the location x, y, z is (0, 0, 0)
    clot_sample_dict["range_clot"] = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the locations
    """

    def __init__(self, top_dict_clot_pickle=None, mode='normal'):

        if top_dict_clot_pickle is None:
            top_dict_clot_pickle = '/data_disk/pulmonary_embolism/simulated_lesions/' \
                                    'clot_sample_list_reduced/volume_range_5%'

        pickle_path_list = extract_all_file_path(top_dict_clot_pickle, end_with='.pickle')

        self.clot_sample_list = []
        print("loading clots...")
        start_time = time.time()

        for pickle_path in pickle_path_list:
            print("loading from path:", pickle_path)
            self.clot_sample_list = self.clot_sample_list + Functions.pickle_load_object(pickle_path)
            if mode == 'temp' or mode == 'debug':
                break
        end_time = time.time()

        print("clot loading complete, cost:", end_time - start_time, 's')
        print("there are", len(self.clot_sample_list), "clot_sample")
        self.length = len(self.clot_sample_list)

    def __len__(self):
        return self.length

    def get_clot_sample_list(self, num_clot_sample, target_volume=None, max_trial=np.inf):
        """

        :param num_clot_sample: the length of the return list
        :param target_volume: the range of raw volume of the clot,
        like (2000, 20000), like (1000, np.inf), like (0, 1000)
        :param max_trial:
        :return: [clot_sample_dict, ...]
        """
        assert num_clot_sample >= 0
        return_list = []
        while len(return_list) < num_clot_sample:
            temp_clot = random_select_clot_sample_dict(self.clot_sample_list, target_volume=target_volume,
                                                       max_trial=max_trial, raise_error=True)
            if temp_clot is not None:
                return_list.append(temp_clot)
        return return_list

    def get_batch_clot_sample_list(self, batch_size, num_clot_sample, target_volume=None, max_trial=np.inf):
        """
        :return: [[...], [...], ]
        """
        return_list = []
        for sample_id in range(batch_size):
            return_list.append(self.get_clot_sample_list(num_clot_sample, target_volume, max_trial))
        return return_list


# form a list-like object, each item is (sample, sample_importance)
class SampleDataset:
    """
    Each sample:
    {"center_line_loc_array": , "sample_sequence": , ...,}

    the training and inference only need "sample_sequence", which is a list of dict
    each dict in "sample_sequence":  (cube in float16)
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube,
    'branch_level': float(branch_level_average), 'clot_array': None, "blood_region": blood_cube}
    """

    def __init__(self, sample_dir_list=None,  # directory or list of directory for storing CT sample sequences
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 sample_interval=(0, 1),
                 wrong_file_name=None,  # the list of file names to remove
                 sample_importance_dict=None,  # {"name": importance (float)}
                 shuffle_path_list=False,
                 func_get_importance=None,
                 ):
        assert sample_dir_list is not None
        assert mode in ['train', 'test']
        if type(sample_dir_list) is str:
            sample_dir_list = [sample_dir_list]
        sample_path_list = []
        if wrong_file_name is not None:
            wrong_file_name = list(wrong_file_name)
        else:
            wrong_file_name = []

        if sample_importance_dict is not None:
            name_set_with_importance = set(sample_importance_dict.keys())
        else:
            name_set_with_importance = set()

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

                if name in name_set_with_importance:
                    sample_importance = sample_importance_dict[name]
                elif name[:-7] in name_set_with_importance:
                    sample_importance = sample_importance_dict[name[:-7]]
                else:
                    sample_importance = None

                if mode == 'train':
                    if ord_sum % 5 == test_id:
                        continue
                    else:
                        sample_path_list.append((os.path.join(sample_dir, name), sample_importance))
                else:
                    if ord_sum % 5 == test_id:
                        sample_path_list.append((os.path.join(sample_dir, name), sample_importance))
                    else:
                        continue

        for current_sample_dir in sample_dir_list:
            print("getting sample path from:", current_sample_dir)
            process_one_sample_dir(current_sample_dir)

        if shuffle_path_list:
            random.shuffle(sample_path_list)

        self.sample_path_list = sample_path_list[sample_interval[0]:: sample_interval[1]]
        self.length = len(self.sample_path_list)
        print("there are", self.length, "samples under:", sample_dir_list)
        print("loading...")
        start_time = time.time()
        self.sample_list = []
        loaded_count = 0
        for idx in range(len(self.sample_path_list)):
            sample, importance = \
                Functions.pickle_load_object(self.sample_path_list[idx][0]), self.sample_path_list[idx][1]
            if importance is None:
                if func_get_importance is None:
                    importance = self.default_func_sample_importance(sample)
                else:
                    importance = func_get_importance(sample)
            self.sample_list.append((sample, importance))
            loaded_count += 1
            if loaded_count % 1000 == 0 and loaded_count > 0:
                print(loaded_count, '/', self.length)

        end_time = time.time()
        print("original sample loaded, cost:", end_time - start_time, 's')
        self.pointer = 0
        self.iter_pointer = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample_list[idx % self.length]  # (sample, importance score)

    def get_item(self):
        return_value = self.sample_list[self.pointer % self.length]
        self.pointer += 1
        return return_value  # (sample, importance score)

    def random_get_item(self):
        return self.sample_list[random.randint(0, self.length - 1)]

    def get_batch(self, batch_size, random_select=False, key_on_sample=None):
        sample_list = []
        importance_score_list = []
        for i in range(batch_size):
            if not random_select:
                sample, importance_score = self.get_item()
            else:
                sample, importance_score = self.random_get_item()
            if key_on_sample is not None:
                sample = sample[key_on_sample]
            sample_list.append(sample)
            importance_score_list.append(importance_score)
        return sample_list, importance_score_list

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

    @staticmethod
    def default_func_sample_importance(sample):
        """
        sample importance:
        1 for non-PE, 1 for PE good register good pair, 1.5 for PE perfect register good pair
        2.5 for PE good register perfect pair, 3 for PE perfect register, perfect pair

        :param sample:
        :return: float
        """

        key_set = sample.keys()

        if "relative_importance" in key_set:
            return sample["relative_importance"]

        if not sample['is_PE']:  # from not PE non-contrast:
            return 1.

        assert sample['has_clot_gt']
        if sample['registration_quality'] == 'good' and sample['pe_pair_quality'] == 'good':
            return 1.

        if sample['registration_quality'] == 'perfect' and sample['pe_pair_quality'] == 'good':
            return 1.5

        if sample['registration_quality'] == 'good' and sample['pe_pair_quality'] == 'perfect':
            return 2.5

        if sample['registration_quality'] == 'perfect' and sample['pe_pair_quality'] == 'perfect':
            return 3.

        raise ValueError

    @staticmethod
    def check_whether_pe(sample):
        return sample['is_PE']

    def check_key_value_for_all_sample(self, key, value):
        for sample in self.sample_list:
            assert sample[key] == value


# the dataloader of non-PE dataset during training or testing
class DataLoaderSimulatedClot:
    """
    Iterative object, prepare data tensors ready for model. Each step return:

    array_packages, list_sample_importance

    the training data is dynamically generated, while the testing data is fixed until updating clot simulate parameters

    """

    def __init__(self, clot_dataset, non_pe_sample_dataset, batch_size, shuffle=False, num_workers=16,
                 show=True, mode='train', num_prepared_dataset_test=3, clot_volume_range=(3000, 30000), min_clot=300,
                 num_clot_each_sample_range=(0, 5), augment=True, embed_dim=None, sample_sequence_length=None):
        """

        :param clot_dataset: instance of ClotDataset:
        :param non_pe_sample_dataset: instance of SampleDataset
        :param batch_size: batch_size during training or testing
        :param shuffle: shuffle the order of sample
        :param num_workers: num cpu when generating samples
        :param show:
        :param mode: training data is dynamically generated, while the testing data is fixed until
        updating clot simulate parameters
        :param num_prepared_dataset_test: int, number times duplicate original sample, and then
        applying clots independently
        :param clot_volume_range: volume range for each clot seed selected
        :param min_clot: when apply clot seed, the volume will reduce, here give the lower bound for final clot volume
        for each applied clot
        :param num_clot_each_sample_range: we can apply multiple clot on each sample
        :param sample_sequence_length: None for adaptive, optimize GPU ram but unknown tensor shapes
                                      you can specify like 4000 for high resolution, 1500 for low resolution
        """
        self.sample_sequence_length = sample_sequence_length
        self.embed_dim = embed_dim
        assert embed_dim is not None
        self.augment = augment
        assert mode in ['train', 'test']
        if show:
            print("mode:", mode)
            print("batch_size:", batch_size, "shuffle:", shuffle, "num_workers:", num_workers,
                  "num_prepared_dataset_test:", num_prepared_dataset_test, "clot_volume_range:", clot_volume_range,
                  "min_clot:", min_clot, "num_clot_each_sample:", num_clot_each_sample_range)

        for (sample, sample_importance) in non_pe_sample_dataset:
            assert not sample['is_PE']

        self.mode = mode
        self.clot_dataset = clot_dataset
        self.non_pe_sample_dataset = non_pe_sample_dataset
        self.epoch_passed = 0
        self.num_prepared_dataset_test = num_prepared_dataset_test
        self.clot_volume_range = clot_volume_range
        self.min_clot = min_clot
        self.num_clot_each_sample_range = num_clot_each_sample_range

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.sample_loaded = 0
        self.num_ct = len(non_pe_sample_dataset)
        assert self.num_ct > batch_size

        self.num_clot_seed = len(clot_dataset)
        self.current_clot_seed = 0

        self.add_clot_on_sample = partial(apply_clot_on_sample, func_change_ct=self.func_change_ct,
                                          power_range=(-0.3, 0.6), add_base_range=(0, 3),
                                          value_increase=(0.1, 0.2), voxel_variance=(0.5, 1),
                                          min_volume=self.min_clot, max_trial=10, augment=augment, visualize=False)

        self.receive_end_list = []
        self.sub_process_list = []
        self.sub_process_receive_pointer = 0  # point at the oldest sub_sequence

        self.prepared_test_dataset = None  # [(array_packages, list_sample_importance), ...]
        self.train_batch_passed = 0  # to determine when stop iteration
        self.test_batch_pointer = 0  # used to extract data from prepared_test_dataset

        self.num_batch_processed = 0

    @staticmethod
    def func_change_ct(clot_depth, add_base, power):
        return (clot_depth + add_base) ** power

    def update_clot_simulation_parameter(self, power_range, add_base_range, value_increase, voxel_variance,
                                         min_clot=None, max_trial=10, augment=None):
        print("\n\n#########################################################")
        self.prepared_test_dataset = None  # reset test dataset
        if min_clot is not None:
            self.min_clot = min_clot
        if augment is not None:
            assert augment in [True, False]
            self.augment = augment
        print(self.mode, "dataloader updating simulation parameters:")
        print("power_range:", power_range, "add_base_range:", add_base_range, "value_increase:", value_increase,
              "voxel_variance:", voxel_variance, "augment:", self.augment, "min_clot:", self.min_clot)
        print("#########################################################")
        self.add_clot_on_sample = partial(apply_clot_on_sample, func_change_ct=self.func_change_ct,
                                          power_range=power_range, add_base_range=add_base_range,
                                          value_increase=value_increase, voxel_variance=voxel_variance,
                                          min_volume=self.min_clot, max_trial=max_trial, augment=self.augment,
                                          visualize=False)

    def get_input_list_for_sub_process_func(self, send_end):
        """
        :return: [send_end, [(sample, list_clot_sample_dict, sample_importance), ]]
        """
        num_clot = random.randint(self.num_clot_each_sample_range[0], self.num_clot_each_sample_range[1])
        value_list = []
        for i in range(self.batch_size):
            sample, sample_importance = self.non_pe_sample_dataset.get_item()
            list_clot_sample_dict = self.clot_dataset.get_clot_sample_list(
                num_clot, target_volume=self.clot_volume_range, max_trial=np.inf)
            value_list.append((sample, list_clot_sample_dict, sample_importance))
        return send_end, value_list

    def sub_process_func(self, *input_list):
        """
        each sub process return (array_packages, list_sample_importance).
        its length is the batch_size. item is sample sequence with clot, True/False, respectively
        :param input_list: [send_end, [(sample, list_clot_sample_dict, sample_importance), ]]
        """
        send_end, value_list = input_list
        list_sample_sequence = []
        list_sample_importance = []

        for sample, list_clot_sample_dict, sample_importance in value_list:
            list_sample_sequence.append(self.add_clot_on_sample(sample, list_clot_sample_dict))
            list_sample_importance.append(sample_importance)

        array_packages = prepare_tensors_simulate_clot(list_sample_sequence, self.embed_dim, device=None,
                                                       training_phase=True,
                                                       sample_sequence_len=self.sample_sequence_length)

        send_end.send((array_packages, list_sample_importance))  # send speed is around 10-100 MB/s
        send_end.close()

    def establish_new_sub_process(self):
        receive_end, send_end = multiprocessing.Pipe(duplex=False)
        input_list = self.get_input_list_for_sub_process_func(send_end)
        sub_process = multiprocessing.Process(target=self.sub_process_func, args=input_list)
        sub_process.start()
        return receive_end, sub_process

    def establish_initial_sub_process_queue(self):
        """
        establish the queue for sub_processes
        """
        print("establishing sub process queue")
        assert len(self.receive_end_list) == 0 and len(self.sub_process_list) == 0
        for i in range(self.num_workers):
            receive_end, sub_process = self.establish_new_sub_process()
            self.receive_end_list.append(receive_end)
            self.sub_process_list.append(sub_process)

    def whether_need_a_new_sub_process(self):
        running_sub_process = 0
        for sub_process in self.sub_process_list:
            if sub_process is not None:
                running_sub_process += 1
        if self.num_batch_processed <= len(self) - running_sub_process:
            return True
        return False

    def extract_data_from_sub_process_and_start_a_new(self):
        if not (len(self.receive_end_list) > 0 and len(self.sub_process_list) > 0):
            self.establish_initial_sub_process_queue()

        sub_process = self.sub_process_list[self.sub_process_receive_pointer]
        receive_end = self.receive_end_list[self.sub_process_receive_pointer]

        # receive data from the oldest sub process
        array_packages, list_sample_importance = receive_end.recv()
        self.num_batch_processed += 1

        # some operation for sub_process if it goes wrong (Optional)
        sub_process.join()
        sub_process.terminate()

        # start a new sub process when necessary
        if self.whether_need_a_new_sub_process():
            receive_end, sub_process = self.establish_new_sub_process()
        else:
            receive_end, sub_process = None, None
        self.sub_process_list[self.sub_process_receive_pointer] = sub_process
        self.receive_end_list[self.sub_process_receive_pointer] = receive_end

        # update pointer
        self.sub_process_receive_pointer += 1
        self.sub_process_receive_pointer = self.sub_process_receive_pointer % self.num_workers

        return array_packages, list_sample_importance

    def clear_sub_process_queue(self):
        # StopIteration() will join sub process.
        # If send data for sub process exceed 64 KB, automatic join will failed
        for sub_process in self.sub_process_list:
            if sub_process is not None:
                sub_process.kill()
        self.sub_process_list = []
        self.receive_end_list = []

    def preparing_testing_dataset(self):
        print("preparing new test dataset")
        start_time = time.time()
        list_of_test_batch = []
        while len(list_of_test_batch) < len(self):
            array_packages, list_sample_importance = self.extract_data_from_sub_process_and_start_a_new()
            list_of_test_batch.append((array_packages, list_sample_importance))
        self.prepared_test_dataset = list_of_test_batch
        print("test dataset prepared. cost:", time.time() - start_time, 's')
        print("num batches in test dataset:", len(self.prepared_test_dataset))

    def __len__(self):
        if self.mode == 'train':
            samples_num = self.num_ct
        else:
            samples_num = int(self.num_ct * self.num_prepared_dataset_test)
        if samples_num % self.batch_size == 0:
            return int(samples_num / self.batch_size)
        return int(samples_num / self.batch_size) + 1

    def __iter__(self):
        print("\n\n#########################################################")
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)
        print("#########################################################")
        if self.shuffle:
            self.non_pe_sample_dataset.shuffle()

        if self.mode == 'test' and self.prepared_test_dataset is None:
            self.num_batch_processed = 0
            self.preparing_testing_dataset()

        if self.mode == 'train':
            self.num_batch_processed = 0

        self.test_batch_pointer = 0
        self.train_batch_passed = 0

        return self

    def check_stop_iteration(self):
        if self.mode == 'train':
            if self.train_batch_passed >= len(self):
                self.epoch_passed += 1
                self.clear_sub_process_queue()
                raise StopIteration()
        else:
            if self.test_batch_pointer >= len(self.prepared_test_dataset):
                self.epoch_passed += 1
                self.clear_sub_process_queue()
                raise StopIteration()

    def __next__(self):
        self.check_stop_iteration()
        if self.mode == 'train':
            array_packages, list_sample_importance = self.extract_data_from_sub_process_and_start_a_new()
            self.train_batch_passed += 1
        else:
            array_packages, list_sample_importance = self.prepared_test_dataset[self.test_batch_pointer]
            self.test_batch_pointer += 1
        return array_packages, list_sample_importance


# the dataloader of PE dataset during training or testing, list like object
class DataLoaderWithAnnotation:
    def __init__(self, pe_sample_dataset, batch_size, num_workers=6, augment=True, random_select=False, shuffle=True,
                 embed_dim=None, sample_sequence_length=None):
        """

        :param pe_sample_dataset: instance of SampleDataset
        :param batch_size:
        :param num_workers:
        :param augment:
        :param random_select:
        :param shuffle:
        :param embed_dim:
        :param sample_sequence_length:
        """
        self.pe_sample_dataset = pe_sample_dataset
        self.batch_size = batch_size
        self.random_select = random_select
        self.sample_sequence_length = sample_sequence_length
        self.embed_dim = embed_dim
        self.num_workers = num_workers
        self.augment = augment
        self.shuffle = shuffle

        if shuffle:
            self.pe_sample_dataset.shuffle()

        self.receive_end_list = []
        self.sub_process_list = []
        self.sub_process_receive_pointer = 0  # point at the oldest sub_sequence

    def get_input_list_for_sub_process_func(self, send_end):
        """
        :return: [send_end, list_sample, list_sample_importance]
        """
        list_sample, list_sample_importance = self.pe_sample_dataset.get_batch(
            self.batch_size, random_select=self.random_select, key_on_sample=None)
        return send_end, list_sample, list_sample_importance

    def sub_process_func(self, *input_list):
        """
        each sub process return (array_packages, list_sample_importance).
        :param input_list: [send_end, list_sample, list_sample_importance]
        """
        send_end, list_sample, list_sample_importance = input_list

        list_sample_sequence = []
        for sample in list_sample:
            if self.augment:
                sample_augmented = random_flip_rotate_swap_sample(sample, labels=get_labels(swap_axis=False))
                list_sample_sequence.append(sample_augmented["sample_sequence"])
            else:
                list_sample_sequence.append(sample["sample_sequence"])

        array_packages = prepare_tensors_with_annotation(list_sample_sequence, self.embed_dim, device=None,
                                                         training_phase=True,
                                                         sample_sequence_len=self.sample_sequence_length)

        send_end.send((array_packages, list_sample_importance))  # send speed is around 10-100 MB/s
        send_end.close()

    def establish_new_sub_process(self):
        receive_end, send_end = multiprocessing.Pipe(duplex=False)
        input_list = self.get_input_list_for_sub_process_func(send_end)
        sub_process = multiprocessing.Process(target=self.sub_process_func, args=input_list)
        sub_process.start()
        return receive_end, sub_process

    def establish_initial_sub_process_queue(self):
        """
        establish the queue for sub_processes
        """
        print("establishing sub process queue")
        assert len(self.receive_end_list) == 0 and len(self.sub_process_list) == 0
        if self.shuffle:
            self.pe_sample_dataset.shuffle()
        for i in range(self.num_workers):
            receive_end, sub_process = self.establish_new_sub_process()
            self.receive_end_list.append(receive_end)
            self.sub_process_list.append(sub_process)

    def extract_data_from_sub_process_and_start_a_new(self, start_new=True):
        if self.batch_size == 0:
            return None, None

        if not (len(self.receive_end_list) > 0 and len(self.sub_process_list) > 0):
            self.establish_initial_sub_process_queue()

        sub_process = self.sub_process_list[self.sub_process_receive_pointer]
        receive_end = self.receive_end_list[self.sub_process_receive_pointer]

        # receive data from the oldest sub process
        array_packages, list_sample_importance = receive_end.recv()

        # some operation for sub_process if it goes wrong (Optional)
        sub_process.join()
        sub_process.terminate()

        if start_new:
            # start a new sub process
            receive_end, sub_process = self.establish_new_sub_process()
        else:
            receive_end, sub_process = None, None

        self.sub_process_list[self.sub_process_receive_pointer] = sub_process
        self.receive_end_list[self.sub_process_receive_pointer] = receive_end

        # update pointer
        self.sub_process_receive_pointer += 1
        self.sub_process_receive_pointer = self.sub_process_receive_pointer % self.num_workers

        return array_packages, list_sample_importance

    def clear_sub_process_queue(self):
        # StopIteration() will join sub process.
        # If send data for sub process exceed 64 KB, automatic join will failed
        for sub_process in self.sub_process_list:
            if sub_process is not None:
                sub_process.kill()
        self.sub_process_list = []
        self.receive_end_list = []

    def shuffle_dataset(self):
        self.pe_sample_dataset.shuffle()


def merge_tensor_packages(package_a, package_b):
    """

    :param package_a: (batch_tensor 0, pos_embed_tensor 1, given_vector 2,
    flatten_roi 3, cube_shape 4, clot_gt_tensor 5, penalty_weight_tensor 6), list_sample_importance
    :param package_b:
    :return:
    """
    if package_b == (None, None):
        return package_a
    if package_a == (None, None):
        return package_b
    list_sample_importance = package_a[1] + package_b[1]
    batch_tensor = torch.cat((package_a[0][0], package_b[0][0]), dim=0)
    pos_embed_tensor = torch.cat((package_a[0][1], package_b[0][1]), dim=0)
    if package_a[0][2] is None:
        given_vector = None
    else:
        given_vector = torch.cat((package_a[0][2], package_b[0][2]), dim=0)
    flatten_roi = torch.cat((package_a[0][3], package_b[0][3]), dim=0)
    assert package_a[0][4] == package_a[0][4]
    cube_shape = package_a[0][4]
    if package_a[0][5] is None:
        clot_gt_tensor = None
    else:
        clot_gt_tensor = torch.cat((package_a[0][5], package_b[0][5]), dim=0)
    if package_a[0][6] is None:
        penalty_weight_tensor = None
    else:
        penalty_weight_tensor = torch.cat((package_a[0][6], package_b[0][6]), dim=0)

    return (batch_tensor, pos_embed_tensor, given_vector, flatten_roi,
            cube_shape, clot_gt_tensor, penalty_weight_tensor), list_sample_importance


if __name__ == '__main__':
    sample_dataset = SampleDataset(
        ['/data_disk/pulmonary_embolism_final/training_samples_with_annotation/low_resolution/pe_ready_denoise'],
        sample_interval=(0, 4))

    for sample_, importance_ in sample_dataset:
        print(importance_, list(sample_.keys()))
    exit()
    ClotDataset()
