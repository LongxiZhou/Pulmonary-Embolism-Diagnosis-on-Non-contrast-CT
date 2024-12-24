import os
import random
import Tool_Functions.Functions as Functions
from pulmonary_embolism_v2.simulate_lesion.simulate_clot import apply_clot_on_sample_sequence
from collections import defaultdict
from functools import partial
import time


class PEDatasetSimulateClot:
    """
    Each sample is a .pickle file: a list of dict
    """
    def __init__(self, sample_dir,  # directory for storing CT sample sequences
                 top_dict_clot_seeds,  # directory for storing list-clot_sample_dict
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 wrong_file_name=None,  # the list of file names to remove
                 important_file_name=None
                 ):
        sample_path_list = []
        if important_file_name is None:
            important_file_name = []
        name_list_all_samples = os.listdir(sample_dir)
        if wrong_file_name is not None:
            wrong_file_name = list(wrong_file_name)
            removed_name_list = []
            for name in wrong_file_name:
                if name in name_list_all_samples:
                    print("remove_wrong_file:", name)
                    name_list_all_samples.remove(name)
                    removed_name_list.append(name)
            for name in removed_name_list:
                wrong_file_name.remove(name)
            if len(wrong_file_name) > 0:
                print("these wrong file name is not in the dataset:")
                print(wrong_file_name)

        assert mode in ['train', 'test']

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

        self.sample_path_list = sample_path_list
        self.length = len(self.sample_path_list)

        if os.path.isfile(top_dict_clot_seeds):
            clot_seed_path_list = [top_dict_clot_seeds]
        else:
            clot_seed_path_list = []
            name_list_all_seeds = os.listdir(top_dict_clot_seeds)
            for name in name_list_all_seeds:
                clot_seed_path_list.append(os.path.join(top_dict_clot_seeds, name))

        self.clot_seed_path_list = clot_seed_path_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length
        return self.sample_path_list[idx]  # (path for the pickle object, whether it is important)


class DataLoaderSimulatedClot:

    def __init__(self, dataset, batch_ct, shuffle=False, num_workers=32, drop_last=True,
                 show=True, mode='train', num_prepared_dataset_train=3, num_prepared_dataset_test=5, reuse_count=5,
                 min_clot_count=1000):
        """

        :param dataset: object like "PEDatasetSimulateClot"
        :param batch_ct: in each batch, how many sample sequence?
        :param shuffle:
        :param num_workers: number CPU during preparing the data
        :param drop_last: cut the last few sample that less than batch size
        :param show:
        :param num_prepared_dataset_train: for each sample sequence in the train dataset, we simulate a clot, thus forms
        a "prepared_dataset_train"
        :param num_prepared_dataset_test: for each sample sequence in the test dataset, we simulate a clot, thus forms
        a "prepared_dataset_test"
        :param reuse_count: reuse count for the "prepared_dataset_train" during training
        :param min_clot_count: the least number of clot voxels in each simulated clot
        """
        assert mode in ['train', 'test']

        if show:
            print("mode:", mode)
            print("batch_ct:", batch_ct, "shuffle:", shuffle, "num_workers:", num_workers, "drop_last:", drop_last,
                  "num_prepared_dataset_train:", num_prepared_dataset_train,
                  "num_prepared_dataset_test:", num_prepared_dataset_test, "reuse_count:", reuse_count,
                  "min_clot_count:", min_clot_count)

        self.mode = mode
        self.dataset = dataset
        self.epoch_passed = 0
        self.epoch_passed_prepared_dataset = 0
        self.reuse_count = reuse_count
        self.num_prepared_dataset_test = num_prepared_dataset_test
        self.num_prepared_dataset_train = num_prepared_dataset_train
        self.min_clot_count = min_clot_count

        self.list_prepared_training_dataset = []
        self.prepared_testing_dataset = None

        self.batch_ct = batch_ct
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.sample_loaded = 0
        self.num_ct = len(dataset)
        assert self.num_ct > batch_ct

        self.list_sample_sequence = []

        if show:
            print("loading sample sequences...")
        for idx in range(len(dataset)):
            sample_sequence = Functions.pickle_load_object(self.dataset[idx][0])
            sample_sequence[0]["whether_important"] = self.dataset[idx][1]
            self.list_sample_sequence.append(sample_sequence)
        if show:
            print("There are:", len(self.dataset), "sample sequences")

        self.list_clot_sample_dict = []

        for clot_seed_path in dataset.clot_seed_path_list:
            if show:
                print("loading clot seed at", clot_seed_path)
            self.list_clot_sample_dict = self.list_clot_sample_dict + Functions.pickle_load_object(clot_seed_path)
        if show:
            print("there are", len(self.list_clot_sample_dict), "clot types")

        self.num_clot_samples = len(self.list_clot_sample_dict)
        self.current_clot_sample = 0

        self.show = show
        self.current_dataset = None
        self.current_dataset_len = None

        self.add_clot_on_sequence = partial(apply_clot_on_sample_sequence, func_change_ct=self.func_change_ct,
                                            power_range=(-0.3, 0.6), add_base_range=(0, 3),
                                            value_increase=(0.01, 0.02), voxel_variance=(0.5, 1))

        self.updated_simulation_parameter = False

    def update_clot_simulation_parameter(self, power_range, add_base_range, value_increase, voxel_variance):
        print(self.mode, "dataloader updating simulation parameters:")
        print("power_range:", power_range, "add_base_range:", add_base_range, "value_increase:", value_increase,
              "voxel_variance:", voxel_variance)
        self.add_clot_on_sequence = partial(apply_clot_on_sample_sequence, func_change_ct=self.func_change_ct,
                                            power_range=power_range, add_base_range=add_base_range,
                                            value_increase=value_increase, voxel_variance=voxel_variance)

    def __len__(self):
        if self.mode == 'train':
            if self.drop_last:
                return int(self.num_ct / self.batch_ct)
            else:
                if self.num_ct % self.batch_ct == 0:
                    return int(self.num_ct / self.batch_ct)
                return int(self.num_ct / self.batch_ct) + 1
        return int(self.num_ct * self.num_prepared_dataset_test / self.batch_ct)

    def __iter__(self):
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)
        print("epoch passed for current", self.mode, "dataset:", self.epoch_passed_prepared_dataset)

        if self.mode == 'train' and (self.epoch_passed == 0 or
                                     self.epoch_passed_prepared_dataset >=
                                     self.reuse_count * self.num_prepared_dataset_train):
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

    def __next__(self):

        count_initial = self.sample_loaded
        count_terminal = count_initial + self.batch_ct

        if count_initial >= self.current_dataset_len:
            self.sample_loaded = 0
            self.epoch_passed += 1
            self.epoch_passed_prepared_dataset += 1
            raise StopIteration()

        if count_terminal >= self.current_dataset_len:
            if self.drop_last:
                self.sample_loaded = 0
                self.epoch_passed += 1
                self.epoch_passed_prepared_dataset += 1
                raise StopIteration()
            else:
                count_terminal = self.current_dataset_len

        assert count_initial < self.current_dataset_len

        if count_terminal >= self.current_dataset_len:
            self.sample_loaded = 0
            self.epoch_passed += 1
            self.epoch_passed_prepared_dataset += 1
            raise StopIteration()

        self.sample_loaded = count_terminal

        return self.current_dataset[count_initial: count_terminal]  # a list of (sample_sequences, clot_voxel_count)

    def simulate_clot_on_dataset(self):
        """

        :return: list_sample_sequence_with_clot
        """
        if self.shuffle:
            random.shuffle(self.list_sample_sequence)

        if self.num_workers > 1:
            input_list_parallel = []
            separate_dict = defaultdict(list)

            # separate the sample sequence into self.num_workers for parallel
            time_start = time.time()
            for idx, sample_sequence in enumerate(self.list_sample_sequence):
                item = (sample_sequence, self.list_clot_sample_dict[self.current_clot_sample % self.num_clot_samples])
                self.current_clot_sample += 1
                separate_dict[idx % self.num_workers].append(item)

            for idx in list(separate_dict.keys()):
                input_list_parallel.append((self.add_clot_on_sequence, separate_dict[idx]))
            time_end = time.time()
            print("separate sequence for parallel cost:", time_end - time_start)

            time_start = time.time()
            processed_list = Functions.func_parallel(self.simulate_clot_for_list_sequences, input_list_parallel,
                                                     parallel_count=self.num_workers)
            time_end = time.time()
            print("apply clot cost:", time_end - time_start)

            time_start = time.time()
            list_sample_sequence_with_clot = []
            for sub_list in processed_list:
                list_sample_sequence_with_clot = list_sample_sequence_with_clot + sub_list
            time_end = time.time()
            print("merge list cost:", time_end - time_start)

        else:
            list_sample_sequence_with_clot = []
            time_start = time.time()
            num_samples = len(self.list_sample_sequence)
            processed = 0
            for sample_sequence in self.list_sample_sequence:
                clot_seed = self.list_clot_sample_dict[self.current_clot_sample % self.num_clot_samples]
                sample_sequence_with_clot = self.add_clot_on_sequence(sample_sequence, clot_seed)
                self.current_clot_sample += 1
                list_sample_sequence_with_clot.append(sample_sequence_with_clot)

                processed += 1
                if processed % int(num_samples / 5) == 0:
                    print("processed:", processed, "/", num_samples, "time passed:", time.time() - time_start)

        return list_sample_sequence_with_clot

    def prepare_training_dataset(self):
        assert self.mode == 'train'
        print("establishing new training dataset...")
        del self.list_prepared_training_dataset
        self.list_prepared_training_dataset = []
        for dataset_count in range(self.num_prepared_dataset_train):
            time_start = time.time()
            print(dataset_count, 'out of', self.num_prepared_dataset_train)
            self.list_prepared_training_dataset.append(self.simulate_clot_on_dataset())
            time_end = time.time()
            print("cost time:", time_end - time_start)
        self.epoch_passed_prepared_dataset = 0

    def prepare_testing_dataset(self):
        assert self.mode == 'test'
        print("establishing new testing dataset...")
        time_start = time.time()
        del self.prepared_testing_dataset
        self.prepared_testing_dataset = []
        for dataset_count in range(self.num_prepared_dataset_test):
            self.prepared_testing_dataset = \
                self.prepared_testing_dataset + self.simulate_clot_on_dataset()
        time_end = time.time()
        print("cost time:", time_end - time_start)
        self.epoch_passed_prepared_dataset = 0

    @staticmethod
    def func_change_ct(clot_depth, add_base, power):
        return (clot_depth + add_base) ** power

    def simulate_clot_for_list_sequences(self, input_tuple):
        """

        :param input_tuple: (func_add_clot, [(sample_sequence, clot_sample_dict), ])
        :return: [sample_sequence_with_clot, ]
        """
        func_add_clot = input_tuple[0]

        list_sample_with_clot = []

        for item in input_tuple[1]:
            sample_sequence, clot_sample_dict = item
            sample_sequence_with_clot, num_clot_voxels = func_add_clot(sample_sequence, clot_sample_dict)
            re_calculate_count = 0
            while num_clot_voxels < self.min_clot_count and re_calculate_count < 9:
                sample_sequence_with_clot, num_clot_voxels = func_add_clot(sample_sequence, clot_sample_dict)
                re_calculate_count += 1
            if num_clot_voxels >= self.min_clot_count:
                list_sample_with_clot.append((sample_sequence_with_clot, num_clot_voxels))

        return list_sample_with_clot  # item is (sample_sequence_with_clot, num_clot_voxels)


if __name__ == '__main__':

    exit()
