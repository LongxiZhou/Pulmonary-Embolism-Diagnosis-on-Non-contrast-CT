import os
import random
import Tool_Functions.Functions as Functions
from segment_clot_cta.prepare_training_dataset.simulate_clot_pe_v3 import \
    random_select_clot_sample_dict, apply_clot_on_sample
from pulmonary_embolism_v3.utlis.phase_control_and_sample_process import prepare_tensors_pe_transformer
from functools import partial
from Tool_Functions.file_operations import extract_all_file_path
import multiprocessing
import time
import numpy as np


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
            if mode == 'temp':
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


# form a list-like object, each item is (sample, whether it is important)
class OriginalSampleDataset:
    """
    Each sample:
    {"center_line_loc_array": , "sample_sequence": , "additional_info": ,}

    the training and inference only need "sample_sequence", which is a list of dict
    each dict in "sample_sequence":  (cube in float16)
    {'ct_data': ct_cube, 'penalty_weight': None, 'location_offset': central_location_offset,
    'given_vector': None, 'center_location': central_location, 'depth_cube': depth_cube,
    'branch_level': float(branch_level_average), 'clot_array': None, "blood_region": blood_cube}
    """

    def __init__(self, sample_dir_list=None,  # list of directory for storing CT sample sequences
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 sample_interval=(0, 1),
                 wrong_file_name=None,  # the list of file names to remove
                 important_file_name=None,
                 shuffle_path_list=False
                 ):
        if sample_dir_list is None:
            sample_dir_list = '/data_disk/pulmonary_embolism/segment_clot_on_CTA/' \
                              'PE_CTA_with_gt/sample_sequence/pe_v3/denoise_high-resolution'
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
            self.sample_list.append((Functions.pickle_load_object(self.sample_path_list[idx][0]),
                                     self.sample_path_list[idx][1]))
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
class DataLoaderSimulatedClot:
    """
    Iterative object, prepare data tensors ready for model. Each step return:

    array_packages, list_whether_important

    the training data is dynamically generated, while the testing data is fixed until updating clot simulate parameters

    """

    def __init__(self, clot_dataset, original_sample_dataset, batch_size, shuffle=False, num_workers=8,
                 show=True, mode='train', num_prepared_dataset_test=3, clot_volume_range=(3000, 30000), min_clot=300,
                 num_clot_each_sample_range=(0, 5), augment=True, embed_dim=None, trace_clot=True, roi='blood_vessel',
                 global_bias_range=(-0., 0.)):
        """

        :param clot_dataset: instance of ClotDataset:
        :param original_sample_dataset: instance of OriginalSampleDataset
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
        :param trace_clot: if True, build the clot_volume_vectors to trace different clots
        :param roi: "blood_region" or "blood_vessel"
        :param global_bias_range: the clot region is added a global shift to reduce difficulty.
                in rescaled, i.e., HU value / 1600
        """
        self.global_bias_range = global_bias_range
        self.roi = roi
        self.trace_clot = trace_clot
        self.embed_dim = embed_dim
        assert embed_dim is not None
        self.augment = augment
        assert mode in ['train', 'test']
        if show:
            print("mode:", mode)
            print("batch_size:", batch_size, "shuffle:", shuffle, "num_workers:", num_workers,
                  "num_prepared_dataset_test:", num_prepared_dataset_test, "clot_volume_range:", clot_volume_range,
                  "min_clot:", min_clot, "num_clot_each_sample:", num_clot_each_sample_range)

        self.mode = mode
        self.clot_dataset = clot_dataset
        self.original_sample_dataset = original_sample_dataset
        self.epoch_passed = 0
        self.num_prepared_dataset_test = num_prepared_dataset_test
        self.clot_volume_range = clot_volume_range
        self.min_clot = min_clot
        self.num_clot_each_sample_range = num_clot_each_sample_range

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.sample_loaded = 0
        self.num_ct = len(original_sample_dataset)
        assert self.num_ct > batch_size

        self.num_clot_seed = len(clot_dataset)
        self.current_clot_seed = 0

        self.add_clot_on_sample = partial(apply_clot_on_sample, trace_clot=self.trace_clot,
                                          min_volume=self.min_clot, max_trial=10, augment=augment, visualize=False,
                                          global_bias_range=self.global_bias_range)

        self.receive_end_list = []
        self.sub_process_list = []
        self.sub_process_receive_pointer = 0  # point at the oldest sub_sequence

        self.prepared_test_dataset = None  # [(array_packages, list_whether_important), ...]
        self.train_batch_passed = 0  # to determine when stop iteration
        self.test_batch_pointer = 0  # used to extract data from prepared_test_dataset

        self.num_batch_processed = 0

    def update_clot_simulation_parameter(self, global_bias_range=None, min_clot=None, max_trial=10,
                                         augment=None, trace_clot=None):
        print("\n\n#########################################################")
        print("clear prepared test dataset")
        self.prepared_test_dataset = None  # reset test dataset
        if min_clot is not None:
            self.min_clot = min_clot
        if augment is not None:
            assert augment in [True, False]
            self.augment = augment
        if trace_clot is not None:
            assert trace_clot in [True, False]
            self.trace_clot = trace_clot
        if global_bias_range is not None:
            self.global_bias_range = global_bias_range

        print(self.mode, "dataloader updating simulation parameters:")
        print("global_bias_range:", self.global_bias_range, "augment:", self.augment, "min_clot:",
              self.min_clot, "trace_clot:", self.trace_clot)
        print("#########################################################")
        self.add_clot_on_sample = partial(apply_clot_on_sample, trace_clot=self.trace_clot,
                                          min_volume=self.min_clot, max_trial=max_trial,
                                          augment=self.augment, visualize=False,
                                          global_bias_range=self.global_bias_range)

    def get_input_list_for_sub_process_func(self, send_end):
        """
        :return: [send_end, [(sample, list_clot_sample_dict, whether_important), ]]
        """
        num_clot = random.randint(self.num_clot_each_sample_range[0], self.num_clot_each_sample_range[1])
        value_list = []
        for i in range(self.batch_size):
            sample, whether_important = self.original_sample_dataset.get_item()
            list_clot_sample_dict = self.clot_dataset.get_clot_sample_list(
                num_clot, target_volume=self.clot_volume_range, max_trial=np.inf)
            value_list.append((sample, list_clot_sample_dict, whether_important))
        return send_end, value_list

    def sub_process_func(self, *input_list):
        """
        each sub process return (array_packages, list_whether_important).
        its length is the batch_size. item is sample sequence with clot, True/False, respectively
        :param input_list: [send_end, [(sample, list_clot_sample_dict, whether_important), ]]
        """
        send_end, value_list = input_list
        list_sample_sequence = []
        list_whether_important = []

        for sample, list_clot_sample_dict, whether_important in value_list:
            list_sample_sequence.append(self.add_clot_on_sample(sample, list_clot_sample_dict))
            list_whether_important.append(whether_important)

        array_packages = prepare_tensors_pe_transformer(list_sample_sequence, self.embed_dim, device=None,
                                                        training_phase=True, roi=self.roi, trace_clot=self.trace_clot)

        send_end.send((array_packages, list_whether_important))  # send speed is around 10-100 MB/s
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
        array_packages, list_whether_important = receive_end.recv()
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

        return array_packages, list_whether_important

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
            array_packages, list_whether_important = self.extract_data_from_sub_process_and_start_a_new()
            list_of_test_batch.append((array_packages, list_whether_important))
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
            self.original_sample_dataset.shuffle()

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
            array_packages, list_whether_important = self.extract_data_from_sub_process_and_start_a_new()
            self.train_batch_passed += 1
        else:
            array_packages, list_whether_important = self.prepared_test_dataset[self.test_batch_pointer]
            self.test_batch_pointer += 1
        return array_packages, list_whether_important


if __name__ == '__main__':
    ClotDataset()
