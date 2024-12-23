import numpy as np
import os
import random
import Tool_Functions.Functions as Functions
import multiprocessing as mp


class WeightedChestDatasetTransformer:
    """
    Each sample is a .pickle file: a list of dict
    """
    def __init__(self, sample_dir,  # directory for storing samples
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 wrong_file_name=None  # the list of file names to remove
                 ):
        sample_path_list = []
        name_list_all_samples = os.listdir(sample_dir)
        if wrong_file_name is not None:
            for name in wrong_file_name:
                if name in name_list_all_samples:
                    name_list_all_samples.remove(name)

        assert mode in ['train', 'test']

        for name in name_list_all_samples:
            ord_sum = 0
            for char in name:
                ord_sum += ord(char)
            if mode == 'train':
                if ord_sum % 5 == test_id:
                    continue
                else:
                    sample_path_list.append(os.path.join(sample_dir, name))
            else:
                if ord_sum % 5 == test_id:
                    sample_path_list.append(os.path.join(sample_dir, name))
                else:
                    continue

        self.sample_path_list = sample_path_list

        self.length = len(self.sample_path_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length
        return self.sample_path_list[idx]  # return the path for the pickle object


class WeightedChestDatasetFocal:
    """
        Each sample is a .pickle file: a list of dict
        """

    def __init__(self, sample_dir_a,  # directory for storing samples a, like all_ct
                 sample_dir_b,  # directory for storing samples b, like vessels
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 wrong_file_name=None  # the list of file names to remove
                 ):
        sample_path_list_a = []
        sample_path_list_b = []
        name_list_all_samples = os.listdir(sample_dir_a)

        assert set(name_list_all_samples) == set(os.listdir(sample_dir_b))

        if wrong_file_name is not None:
            for name in wrong_file_name:
                if name in name_list_all_samples:
                    name_list_all_samples.remove(name)

        assert mode in ['train', 'test']

        for name in name_list_all_samples:
            ord_sum = 0
            for char in name:
                ord_sum += ord(char)
            if mode == 'train':
                if ord_sum % 5 == test_id:
                    continue
                else:
                    sample_path_list_a.append(os.path.join(sample_dir_a, name))
                    sample_path_list_b.append(os.path.join(sample_dir_b, name))
            else:
                if ord_sum % 5 == test_id:
                    sample_path_list_a.append(os.path.join(sample_dir_a, name))
                    sample_path_list_b.append(os.path.join(sample_dir_b, name))
                else:
                    continue

        self.sample_path_list_a = sample_path_list_a
        self.sample_path_list_b = sample_path_list_b

        self.length = len(self.sample_path_list_a)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length
        return self.sample_path_list_a[idx], self.sample_path_list_b[idx]  # return the path for the pickle object


def extract_sequence(input_tuple):
    """
    extract content for one ct scan.
    :param input_tuple: (content, ratio_mask_input, ratio_predict, input_output_overlap)
    :return: information_sequence, query_sequence, ct_data_sequence, penalty_array_sequence
    """
    content = input_tuple[0]  # pickle object of a scan, in [dict, dict, dict, ...]
    ratio_mask_input = input_tuple[1]
    ratio_predict = input_tuple[2]
    input_output_overlap = input_tuple[3]

    information_sequence = []
    query_sequence = []
    ct_data_sequence = []
    penalty_array_sequence = []

    num_element = len(content)
    list_element_index = list(np.arange(0, num_element))

    random.shuffle(list_element_index)

    length_information_sequence = int((1 - ratio_mask_input) * num_element)
    length_query_sequence = int(ratio_predict * num_element)

    if not input_output_overlap:
        length_query_sequence = min(length_query_sequence, num_element - length_information_sequence)

    for idx in range(length_information_sequence):
        information_sequence.append(content[list_element_index[idx]])  # here will not deepcopy content[item_id]

    if input_output_overlap:
        random.shuffle(list_element_index)
        for idx in range(length_query_sequence):
            query_sequence.append(content[list_element_index[idx]]['location_offset'])  # a tuple like (x, y, z)
            ct_data_sequence.append(content[list_element_index[idx]]['ct_data'])
            penalty_array_sequence.append(content[list_element_index[idx]]['penalty_weight'])
    else:
        for idx in range(length_information_sequence, length_information_sequence + length_query_sequence):
            query_sequence.append(content[list_element_index[idx]]['location_offset'])
            ct_data_sequence.append(content[list_element_index[idx]]['ct_data'])
            penalty_array_sequence.append(content[list_element_index[idx]]['penalty_weight'])

    return information_sequence, query_sequence, ct_data_sequence, penalty_array_sequence


class PreparedDataset:
    """
        stores the prepared dataset
    """

    def __init__(self, prepared_dataset):
        self.prepared_dataset = prepared_dataset

    def __len__(self):
        return len(self.prepared_dataset["list_sample_sequence"])

    def renew_dataset(self, new_dataset):
        del self.prepared_dataset
        self.prepared_dataset = new_dataset

    def abstract_batch(self, initial_index, terminal_index):
        return_batch = {"list_sample_sequence": self.prepared_dataset["list_sample_sequence"][
                                                     initial_index: terminal_index],
                        "list_query_sequence": self.prepared_dataset["list_query_sequence"][
                                                     initial_index: terminal_index],
                        "list_ct_data_sequence": self.prepared_dataset["list_ct_data_sequence"][
                                                     initial_index: terminal_index],
                        "list_penalty_array_sequence": self.prepared_dataset["list_penalty_array_sequence"][
                                                     initial_index: terminal_index]}
        return return_batch


class DataLoaderPEIte:
    """
    DataLoader for pulmonary embolism: Data in list of dict
    """
    def __init__(self, dataset, batch_ct, batch_split, ratio_mask_input, ratio_predict, input_output_overlap=False,
                 shuffle=False, num_workers=1, drop_last=True, pin_memory=True, show=False, min_parallel_ct=None,
                 mode='train', num_prepared_dataset_train=3, num_prepared_dataset_test=10, reuse_count=5):
        """

        :param dataset: object like "WeightedChestDatasetTransformer"
        :param batch_ct: in each batch, how many CT scans?
        :param batch_split: for each CT scan, split into how many training samples?
        :param ratio_mask_input: ratio for mask the total number of cubes in the sample
        :param ratio_predict: ratio for cubes that need to predict
        :param input_output_overlap: True, query location sequence may overlap with the input cubes
        :param shuffle: whether to shuffle the object "WeightedChestDatasetTransformer"
        :param num_workers: CPU during loading data
        :param drop_last: cut the last few sample that less than batch size
        :param pin_memory: whether first to store data in the cpu memory
        :param show: actively show loading details
        :param min_parallel_ct: batch size == batch_ct * batch_split, if batch size is too small, cannot parallel
        well. Set this too calculate more batch samples each time, and pip out batch size
        """
        self.reuse_count = reuse_count
        self.num_prepared_dataset_test = num_prepared_dataset_test
        self.num_prepared_dataset_train = num_prepared_dataset_train
        self.mode = mode
        assert mode in ['train', 'test']
        self.dataset = dataset
        self.epoch_passed = 0

        self.list_prepared_training_dataset = []
        self.prepared_testing_dataset = None

        self.batch_ct = batch_ct
        self.batch_split = batch_split
        self.ratio_mask_input = ratio_mask_input
        self.ratio_predict = ratio_predict
        self.input_output_overlap = input_output_overlap
        self.shuffle = shuffle
        self.leave_cpu_count = mp.cpu_count() - num_workers
        self.drop_last = drop_last

        idx_list = list(np.arange(0, len(dataset)))
        if shuffle:
            random.shuffle(idx_list)
        self.idx_list = idx_list

        self.ct_extracted = 0
        self.num_ct = len(dataset)
        assert self.num_ct > batch_ct

        self.idx_content_dict = {}  # key is the idx, content is dataset[idx]

        for idx in range(self.num_ct):
            if pin_memory:
                self.idx_content_dict[idx] = Functions.pickle_load_object(self.dataset[idx])
            else:
                self.idx_content_dict[idx] = None

        self.pin_memory = pin_memory
        self.show = show

        if min_parallel_ct is not None:
            self.min_parallel_ct = min_parallel_ct
        else:
            self.min_parallel_ct = len(dataset)

        if show:
            print("There are:", len(self.dataset), "ct scans")

        self.current_dataset = None
        self.current_dataset_len = None

    def __iter__(self):
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)
        if self.mode == 'train' and self.epoch_passed % (self.reuse_count * self.num_prepared_dataset_train) == 0:
            self.prepare_training_dataset()
        if self.mode == 'test' and self.epoch_passed == 0:
            self.prepare_testing_dataset()

        if self.mode == 'train':
            current_dataset = self.list_prepared_training_dataset[self.epoch_passed % self.num_prepared_dataset_train]
        else:
            current_dataset = self.prepared_testing_dataset

        self.current_dataset = current_dataset
        self.current_dataset_len = len(current_dataset)

        return self

    def prepare_training_dataset(self):
        assert self.mode == 'train'
        print("establishing new training dataset...")
        del self.list_prepared_training_dataset
        self.list_prepared_training_dataset = []
        for dataset_count in range(self.num_prepared_dataset_train):
            print(dataset_count, 'out of', self.num_prepared_dataset_train)
            if self.shuffle:
                random.shuffle(self.idx_list)
            prepared_dataset = PreparedDataset(self.extract_batch_list_sequence(self.idx_list))
            self.list_prepared_training_dataset.append(prepared_dataset)

    def prepare_testing_dataset(self):
        assert self.mode == 'test'
        print("establishing new testing dataset...")
        del self.prepared_testing_dataset
        test_id_list = []
        for dataset_count in range(self.num_prepared_dataset_test):
            # print(dataset_count, 'out of', self.num_prepared_dataset_test)
            if self.shuffle:
                random.shuffle(self.idx_list)
            test_id_list = test_id_list + self.idx_list

        self.prepared_testing_dataset = PreparedDataset(self.extract_batch_list_sequence(test_id_list))

    def extract_batch_list_sequence(self, batch_idx_list):

        return_dict_batch = {}

        input_tuple_list = []  # each element is the input of function "extract_sequence"
        # input_tuple: (content, ratio_mask_input, ratio_predict, input_output_overlap)

        for idx in batch_idx_list:
            if self.pin_memory:
                content = self.idx_content_dict[idx]
            else:
                content = Functions.pickle_load_object(self.dataset[idx])

            for duplicate in range(self.batch_split):
                input_tuple_list.append((content, self.ratio_mask_input, self.ratio_predict, self.input_output_overlap))
            """
            here content is a list of dict for a scan, each dict: 
            {'ct_data': ct_cube, 'penalty_weight': penalty_array, 'location_offset': central_location_offset,
                   'given_vector': given_vector}
            """

        list_extracted_tuples = Functions.func_parallel(extract_sequence, input_tuple_list, self.leave_cpu_count)
        # [(information_sequence, query_sequence, ct_data_sequence, penalty_array_sequence), ...]

        packed_batch_list = list(zip(*list_extracted_tuples))

        return_dict_batch["list_sample_sequence"] = list(packed_batch_list[0])
        return_dict_batch["list_query_sequence"] = list(packed_batch_list[1])
        return_dict_batch["list_ct_data_sequence"] = list(packed_batch_list[2])
        return_dict_batch["list_penalty_array_sequence"] = list(packed_batch_list[3])

        if self.show:
            print("The batch size is:", len(packed_batch_list[0]))
            print("The first sample contains a total of:", len(input_tuple_list[0][0]), 'items.')
            print("The first sample contains", len(packed_batch_list[0][0]), "items as information, and",
                  len(packed_batch_list[1][0]), "locations for query.")

        return return_dict_batch

    def __next__(self):

        if self.mode == 'test':
            if self.ct_extracted >= self.num_ct * self.num_prepared_dataset_test:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            count_initial = self.ct_extracted
            count_terminal = count_initial + self.batch_ct

            if count_terminal >= self.num_ct * self.num_prepared_dataset_test:
                if self.drop_last:
                    if self.shuffle:
                        random.shuffle(self.idx_list)
                    self.ct_extracted = 0
                    self.epoch_passed += 1
                    raise StopIteration()
                else:
                    count_terminal = self.num_ct * self.num_prepared_dataset_test

            if self.show:
                print("This batch is from the", count_initial, "-th ct scan, to the", count_terminal, "-th ct scan.")

            batch_initial = count_initial * self.batch_split

            assert batch_initial < self.current_dataset_len

            batch_terminal = batch_initial + (count_terminal - count_initial) * self.batch_split

            if batch_terminal >= self.current_dataset_len:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            return_dict_batch = self.current_dataset.abstract_batch(batch_initial, batch_terminal)

            self.ct_extracted = count_initial + self.batch_ct

            # return_dict_batch:
            # {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
            # "list_penalty_array_sequence":}
            return return_dict_batch
        # mode == 'train'
        else:
            if self.ct_extracted >= self.num_ct:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            count_initial = self.ct_extracted
            count_terminal = count_initial + self.batch_ct

            if count_terminal >= self.num_ct:
                if self.drop_last:
                    if self.shuffle:
                        random.shuffle(self.idx_list)
                    self.ct_extracted = 0
                    self.epoch_passed += 1
                    raise StopIteration()
                else:
                    count_terminal = self.num_ct

            if self.show:
                print("This batch is from the", count_initial, "-th ct scan, to the", count_terminal, "-th ct scan.")

            batch_initial = count_initial * self.batch_split

            assert batch_initial < self.current_dataset_len

            batch_terminal = batch_initial + (count_terminal - count_initial) * self.batch_split

            if batch_terminal >= self.current_dataset_len:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            return_dict_batch = self.current_dataset.abstract_batch(batch_initial, batch_terminal)

            self.ct_extracted = count_initial + self.batch_ct

            # return_dict_batch:
            # {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
            # "list_penalty_array_sequence":}
            return return_dict_batch

    def __len__(self):
        if self.mode == 'train':
            if self.drop_last:
                return int(self.num_ct / self.batch_ct)
            else:
                if self.num_ct % self.batch_ct == 0:
                    return int(self.num_ct / self.batch_ct)
                return int(self.num_ct / self.batch_ct) + 1
        return int(self.num_ct * self.num_prepared_dataset_test / self.batch_ct)


class DataLoaderFocal:
    """
    DataLoader for pulmonary embolism.
    We found the distribution of information cubes has great influence to the accuracy.
    We found you need to let the model_guided to cover all_file location to get accurate prediction.
    So first let model_guided know information under random distribution, then gradually convert to the needed distribution
    """
    def __init__(self, dataset, batch_ct, batch_split, ratio_mask_input, ratio_predict, input_output_overlap=False,
                 shuffle=False, num_workers=1, drop_last=True, pin_memory=True, show=False, min_parallel_ct=None,
                 mode='train', num_prepared_dataset_train=3, num_prepared_dataset_test=3, reuse_count=3,
                 initial_portion=(1, 1), end_portion=(0, 1), ratio_mask_input_end=0.25, ratio_predict_end=0.75,
                 num_interval=20, interval_epoch=90):
        """

        :param dataset: object like "WeightedChestDatasetFocal"
        :param batch_ct: in each batch, how many CT scans?
        :param batch_split: for each CT scan, split into how many training samples?
        :param ratio_mask_input: ratio for mask the total number of cubes in the sample
        :param ratio_predict: ratio for cubes that need to predict
        :param input_output_overlap: True, query location sequence may overlap with the input cubes
        :param shuffle: whether to shuffle the object "WeightedChestDatasetTransformer"
        :param num_workers: CPU during loading data
        :param drop_last: cut the last few sample that less than batch size
        :param pin_memory: whether first to store data in the cpu memory
        :param show: actively show loading details
        :param min_parallel_ct: batch size == batch_ct * batch_split, if batch size is too small, cannot parallel
        well. Set this too calculate more batch samples each time, and pip out batch size
        """
        if input_output_overlap is False:
            assert (1 - ratio_mask_input) + ratio_predict <= 1
            assert (1 - ratio_mask_input_end) + ratio_predict_end <= 1

        assert 0 <= ratio_predict <= 1
        assert 0 <= ratio_mask_input <= 1
        assert max(initial_portion) <= 1 and min(initial_portion) >= 0
        assert max(end_portion) <= 1 and min(end_portion) >= 0

        self.initial_portion = initial_portion
        self.end_portion = end_portion
        self.ratio_mask_input_end = ratio_mask_input_end
        self.ratio_predict_end = ratio_predict_end
        self.num_interval = num_interval
        self.interval_epoch = interval_epoch
        self.total_epoch = self.interval_epoch * self.num_interval

        self.reuse_count = reuse_count
        self.num_prepared_dataset_test = num_prepared_dataset_test
        self.num_prepared_dataset_train = num_prepared_dataset_train
        self.mode = mode
        assert mode in ['train', 'test']
        self.dataset = dataset
        self.epoch_passed = 0

        self.list_prepared_training_dataset = []
        self.prepared_testing_dataset = None

        self.batch_ct = batch_ct
        self.batch_split = batch_split
        self.ratio_mask_input = ratio_mask_input
        self.ratio_predict = ratio_predict
        self.input_output_overlap = input_output_overlap
        self.shuffle = shuffle
        self.leave_cpu_count = mp.cpu_count() - num_workers
        self.drop_last = drop_last

        idx_list = list(np.arange(0, len(dataset)))
        if shuffle:
            random.shuffle(idx_list)
        self.idx_list = idx_list

        self.ct_extracted = 0
        self.num_ct = len(dataset)
        assert self.num_ct > batch_ct

        self.idx_content_dict = {}  # key is the idx, content is two lists

        for idx in range(self.num_ct):
            if pin_memory:
                self.idx_content_dict[idx] = \
                    (Functions.pickle_load_object(self.dataset[idx][0]),
                     Functions.pickle_load_object(self.dataset[idx][1]))
            else:
                self.idx_content_dict[idx] = None

        self.pin_memory = pin_memory
        self.show = show

        if min_parallel_ct is not None:
            self.min_parallel_ct = min_parallel_ct
        else:
            self.min_parallel_ct = len(dataset)

        if show:
            print("There are:", len(self.dataset), "ct scans")

        self.current_dataset = None
        self.current_dataset_len = None

    def __iter__(self):
        if self.total_epoch <= self.epoch_passed:
            print("finished total_epoch")
            exit()
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)
        print(self.total_epoch - self.epoch_passed, 'epochs left for the dataloader')

        if self.mode == 'train' and self.epoch_passed % (self.reuse_count * self.num_prepared_dataset_train) == 0:
            self.prepare_training_dataset()

        if self.mode == 'test' and self.epoch_passed % self.interval_epoch == 0:
            self.prepare_testing_dataset()

        if self.mode == 'train':
            current_dataset = self.list_prepared_training_dataset[self.epoch_passed % self.num_prepared_dataset_train]
        else:
            current_dataset = self.prepared_testing_dataset

        self.current_dataset = current_dataset
        self.current_dataset_len = len(current_dataset)

        return self

    def prepare_training_dataset(self):
        assert self.mode == 'train'
        print("establishing new training dataset...")
        del self.list_prepared_training_dataset
        self.list_prepared_training_dataset = []
        for dataset_count in range(self.num_prepared_dataset_train):
            print(dataset_count, 'out of', self.num_prepared_dataset_train)
            if self.shuffle:
                random.shuffle(self.idx_list)
            prepared_dataset = PreparedDataset(self.extract_batch_list_sequence(self.idx_list))
            self.list_prepared_training_dataset.append(prepared_dataset)

    def prepare_testing_dataset(self):
        assert self.mode == 'test'
        print("establishing new testing dataset...")
        del self.prepared_testing_dataset
        test_id_list = []
        for dataset_count in range(self.num_prepared_dataset_test):
            # print(dataset_count, 'out of', self.num_prepared_dataset_test)
            if self.shuffle:
                random.shuffle(self.idx_list)
            test_id_list = test_id_list + self.idx_list

        self.prepared_testing_dataset = PreparedDataset(self.extract_batch_list_sequence(test_id_list))

    def get_portion(self):
        initial_portion_a, initial_portion_b = self.initial_portion
        end_portion_a, end_portion_b = self.end_portion

        current_portion_a = \
            initial_portion_a + (end_portion_a - initial_portion_a) * (self.epoch_passed / self.total_epoch)
        current_portion_b = \
            initial_portion_b + (end_portion_b - initial_portion_b) * (self.epoch_passed / self.total_epoch)

        return current_portion_a, current_portion_b

    def get_ratio(self):
        initial_ratio_mask = self.ratio_mask_input
        initial_ratio_predict = self.ratio_predict
        end_ratio_mask = self.ratio_mask_input_end
        end_ratio_predict = self.ratio_predict_end

        current_ratio_mask = \
            initial_ratio_mask + (end_ratio_mask - initial_ratio_mask) * (self.epoch_passed / self.total_epoch)
        current_ratio_predict = \
            initial_ratio_predict + (end_ratio_predict - initial_ratio_predict) * (self.epoch_passed / self.total_epoch)

        return current_ratio_mask, current_ratio_predict

    def extract_batch_list_sequence(self, batch_idx_list):

        return_dict_batch = {}

        input_tuple_list = []  # each element is the input of function "extract_sequence"
        # input_tuple: (content, ratio_mask_input, ratio_predict, input_output_overlap)

        current_portion_a, current_portion_b = self.get_portion()
        current_ratio_mask, current_ratio_predict = self.get_ratio()

        print("current portion:", (current_portion_a, current_portion_b))
        print("current ratio:", (current_ratio_mask, current_ratio_predict))

        for idx in batch_idx_list:
            if self.pin_memory:
                content = self.idx_content_dict[idx]  # (list_a, list_b)
            else:
                content = (Functions.pickle_load_object(self.dataset[idx][0]),
                           Functions.pickle_load_object(self.dataset[idx][1]))

            len_list_a, len_list_b = len(content[0]), len(content[1])
            random.shuffle(content[0])
            random.shuffle(content[1])

            content = \
                content[0][0: int(current_portion_a * len_list_a)] + content[1][0: int(current_portion_b * len_list_b)]

            for duplicate in range(self.batch_split):
                input_tuple_list.append((content, current_ratio_mask, current_ratio_predict, self.input_output_overlap))
            """
            here content is a list of dict for a scan, each dict: 
            {'ct_data': ct_cube, 'penalty_weight': penalty_array, 'location_offset': central_location_offset,
                   'given_vector': given_vector}
            """

        list_extracted_tuples = Functions.func_parallel(extract_sequence, input_tuple_list, self.leave_cpu_count)
        # [(information_sequence, query_sequence, ct_data_sequence, penalty_array_sequence), ...]

        packed_batch_list = list(zip(*list_extracted_tuples))

        return_dict_batch["list_sample_sequence"] = list(packed_batch_list[0])
        return_dict_batch["list_query_sequence"] = list(packed_batch_list[1])
        return_dict_batch["list_ct_data_sequence"] = list(packed_batch_list[2])
        return_dict_batch["list_penalty_array_sequence"] = list(packed_batch_list[3])

        if self.show:
            print("The batch size is:", len(packed_batch_list[0]))
            print("The first sample contains a total of:", len(input_tuple_list[0][0]), 'items.')
            print("The first sample contains", len(packed_batch_list[0][0]), "items as information, and",
                  len(packed_batch_list[1][0]), "locations for query.")

        return return_dict_batch

    def __next__(self):

        if self.mode == 'test':
            if self.ct_extracted >= self.num_ct * self.num_prepared_dataset_test:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            count_initial = self.ct_extracted
            count_terminal = count_initial + self.batch_ct

            if count_terminal >= self.num_ct * self.num_prepared_dataset_test:
                if self.drop_last:
                    if self.shuffle:
                        random.shuffle(self.idx_list)
                    self.ct_extracted = 0
                    self.epoch_passed += 1
                    raise StopIteration()
                else:
                    count_terminal = self.num_ct * self.num_prepared_dataset_test

            if self.show:
                print("This batch is from the", count_initial, "-th ct scan, to the", count_terminal, "-th ct scan.")

            batch_initial = count_initial * self.batch_split

            assert batch_initial < self.current_dataset_len

            batch_terminal = batch_initial + (count_terminal - count_initial) * self.batch_split

            if batch_terminal >= self.current_dataset_len:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            return_dict_batch = self.current_dataset.abstract_batch(batch_initial, batch_terminal)

            self.ct_extracted = count_initial + self.batch_ct

            # return_dict_batch:
            # {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
            # "list_penalty_array_sequence":}
            return return_dict_batch
        # mode == 'train'
        else:
            if self.ct_extracted >= self.num_ct:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            count_initial = self.ct_extracted
            count_terminal = count_initial + self.batch_ct

            if count_terminal >= self.num_ct:
                if self.drop_last:
                    if self.shuffle:
                        random.shuffle(self.idx_list)
                    self.ct_extracted = 0
                    self.epoch_passed += 1
                    raise StopIteration()
                else:
                    count_terminal = self.num_ct

            if self.show:
                print("This batch is from the", count_initial, "-th ct scan, to the", count_terminal, "-th ct scan.")

            batch_initial = count_initial * self.batch_split

            assert batch_initial < self.current_dataset_len

            batch_terminal = batch_initial + (count_terminal - count_initial) * self.batch_split

            if batch_terminal >= self.current_dataset_len:
                self.ct_extracted = 0
                self.epoch_passed += 1
                raise StopIteration()

            return_dict_batch = self.current_dataset.abstract_batch(batch_initial, batch_terminal)

            self.ct_extracted = count_initial + self.batch_ct

            # return_dict_batch:
            # {"list_sample_sequence":, "list_query_sequence":, "list_ct_data_sequence":,
            # "list_penalty_array_sequence":}
            return return_dict_batch

    def __len__(self):
        if self.mode == 'train':
            if self.drop_last:
                return int(self.num_ct / self.batch_ct)
            else:
                if self.num_ct % self.batch_ct == 0:
                    return int(self.num_ct / self.batch_ct)
                return int(self.num_ct / self.batch_ct) + 1
        return int(self.num_ct * self.num_prepared_dataset_test / self.batch_ct)


if __name__ == '__main__':

    exit()
