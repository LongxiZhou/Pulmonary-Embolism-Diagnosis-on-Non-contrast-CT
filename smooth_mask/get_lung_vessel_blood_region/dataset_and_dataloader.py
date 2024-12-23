import Tool_Functions.Functions as Functions
import random
from smooth_mask.get_lung_vessel_blood_region.extract_blood_region_and_training_sample import apply_lesion_parallel
import time
import numpy as np
import torch
import multiprocessing
from models.Unet_3D.utlis import random_flip_rotate_swap
import os


def get_list_of_lesion_loc_array(lesion_difficulty_level, lesion_top_dict=None):
    """

    :param lesion_top_dict:
    :param lesion_difficulty_level:
    :return: list of lesion_loc_array
    """
    if lesion_top_dict is None:
        lesion_top_dict = '/data_disk/artery_vein_project/extract_blood_region/lesion_simulation'
    assert type(lesion_difficulty_level) is int and 0 <= lesion_difficulty_level <= 8
    if lesion_difficulty_level == 0:  # do not apply lesion
        return []
    if lesion_difficulty_level == 1:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_500-5000_lv0.pickle'))
    if lesion_difficulty_level == 2:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_500-5000_lv1.pickle'))
    if lesion_difficulty_level == 3:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_500-5000_lv2.pickle'))
    if lesion_difficulty_level == 4:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_nodule_volume>250.pickle'))
    if lesion_difficulty_level == 5:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_direct-extract_volume_5000-50000.pickle'))
    if lesion_difficulty_level == 6:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_5000-50000_lv0.pickle'))
    if lesion_difficulty_level == 7:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_5000-50000_lv1.pickle'))
    if lesion_difficulty_level == 8:
        return Functions.pickle_load_object(
            os.path.join(lesion_top_dict, 'list-of-loc-array_surface-growth_volume_5000-50000_lv2.pickle'))


class SmoothMaskDataset:
    """
    each file is a .pickle file, is a list:
    ((x, y, z), info_channel_0, info_channel_1, info_channel_2, info_channel_3)
    here info for channel is (loc_array_channel_0, value_array_channel_0)
    , with shape [4, x, y, z], here x, y, z % 128 == 0
    channel 0 is the input for the model (raw_vessel_mask)
    channel 1 is the gt for the model output (region_to_remove), vessel_refined = raw_vessel_mask - region_to_remove
    channel 2 is the penalty for false negative (fn):
    more root more penalty, total_penalty = 100 * sqrt(np.sum(region_to_remove)),
    the average fn penalty is from 0.15 to 0.5 for different scans
    channel 3 is the penalty for false positive (fp):
    background penalty same with the average fn penalty; total vessel region fp penalty is twice the total fn penalty.
    see functions "get_input_output_and_penalty_array" in this module
    """

    def __init__(self, sample_dir_or_list_of_sample_dir,
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use ord_sum % 5 == test_id as the test samples
                 wrong_file_name=None,
                 sample_interval=(0, 1), num_workers=1):
        """

        :param sample_dir_or_list_of_sample_dir:
        :param mode:
        :param test_id:
        :param wrong_file_name:
        :param sample_interval: if CPU RAM is not enough, reduce dataset size
        """

        assert mode in ['train', 'test']
        if wrong_file_name is None:
            wrong_file_name = set()
        self._wrong_file_name = wrong_file_name
        self._test_id = test_id
        self._mode = mode

        sample_path_list = []

        if type(sample_dir_or_list_of_sample_dir) is str:
            sample_dir = sample_dir_or_list_of_sample_dir
            for name in os.listdir(sample_dir)[sample_interval[0]:: sample_interval[1]]:
                if self.include_sample(name):
                    sample_path_list.append(os.path.join(sample_dir, name))
        else:
            for sample_dir in sample_dir_or_list_of_sample_dir:
                for name in os.listdir(sample_dir)[sample_interval[0]:: sample_interval[1]]:
                    if self.include_sample(name):
                        sample_path_list.append(os.path.join(sample_dir, name))

        self._sample_path_list = sample_path_list
        self._length = len(self._sample_path_list)
        self._num_workers = num_workers
        self.loaded_sample_list = []
        self.load_sample()
        assert self._length == len(self.loaded_sample_list)

    def check_whether_good_scan(self, name):
        if name not in self._wrong_file_name:
            return True
        print(name, 'is a bad scan')
        return False

    def shuffle_dataset(self):
        random.shuffle(self.loaded_sample_list)

    def include_sample(self, name):
        if self.check_whether_good_scan(name):
            ord_sum = 0
            for char in name:
                ord_sum += ord(char)
            if self._mode == 'train':
                if not ord_sum % 5 == self._test_id:
                    return True
            else:
                if ord_sum % 5 == self._test_id:
                    return True
        return False

    def loading_sample_one_thread(self, fold=(0, 1)):
        sub_sample_list = []
        for sample_path in self._sample_path_list[fold[0]:: fold[1]]:
            sub_sample_list.append(np.load(sample_path)['array'])
        return sub_sample_list

    def load_sample(self):
        start_time = time.time()
        sample_list = []
        fold_list = []

        if self._num_workers == 1:
            sample_list = self.loading_sample_one_thread((0, 1))
        else:
            num_cubes_one_thread = 64  # there are RAM limit for thread
            num_sub_list = int(self._length / num_cubes_one_thread) + 1

            for fold in range(num_sub_list):
                fold_list.append((fold, num_sub_list))
            for sub_sample_list in Functions.func_parallel(self.loading_sample_one_thread, fold_list,
                                                           parallel_count=self._num_workers):
                sample_list = sample_list + sub_sample_list

        self.loaded_sample_list = sample_list
        end_time = time.time()
        print("Loaded", len(sample_list), self._mode, 'data to RAM, time spend', end_time - start_time, 'sec')

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        assert 0 <= idx < self._length
        return self.loaded_sample_list[idx]  # float32 numpy array in [4, 128, 128, 128] or [4, 256, 256, 256]


class SimulateLesionDataset:
    """
    for loading different lesion dataset
    """
    def __init__(self, difficulty_level, iterative_loading=True, shuffle=True,
                 lesion_top_dict='/data_disk/artery_vein_project/extract_blood_region/lesion_simulation',
                 penalty_range=(0.15, 0.5)):
        """

        :param difficulty_level:
        :param iterative_loading: True: load all lesions smaller than given difficulty. False: load certain difficulty
        :param shuffle:
        :param lesion_top_dict: stores several .pickles files of lesions
        each .pickle file is a list object, in the list, each item is a loc_array, i.e., return of np.where,
        indication the lesion locations. The mass center for lesions is very close to zero
        :param penalty_range: for non lesion voxels, the penalty average ranges from 0.15 to 0.5

        """
        assert type(difficulty_level) is int and difficulty_level >= 0
        self.difficulty_level = difficulty_level

        if iterative_loading:
            self.list_lesion_loc_array = []
            while difficulty_level > 0:
                self.list_lesion_loc_array = \
                    self.list_lesion_loc_array + get_list_of_lesion_loc_array(difficulty_level, lesion_top_dict)
                difficulty_level -= 1
        else:
            self.list_lesion_loc_array = get_list_of_lesion_loc_array(difficulty_level, lesion_top_dict)

        if shuffle:
            random.shuffle(self.list_lesion_loc_array)

        self.num_lesions = len(self.list_lesion_loc_array)
        print("Lesion dataset of difficulty level:", self.difficulty_level, "contains:", self.num_lesions, 'lesions')

        self.max_lesion_volume = 0
        self.min_lesion_volume = np.inf
        if self.num_lesions > 0:
            for lesion_loc_array in self.list_lesion_loc_array:
                if len(lesion_loc_array[0]) > self.max_lesion_volume:
                    self.max_lesion_volume = len(lesion_loc_array[0])
                if len(lesion_loc_array[0]) < self.min_lesion_volume:
                    self.min_lesion_volume = len(lesion_loc_array[0])
        if self.num_lesions > 0:
            print("Max lesion voxel:", self.max_lesion_volume)
            print("Min lesion voxel:", self.min_lesion_volume)

        self.penalty_range = penalty_range

    def get_arbitrary_lesion(self):
        if self.num_lesions == 0:
            return None
        # by default "no lesion" is 1000 cases
        if random.uniform(0, 1) < 1000 / (1000 + self.num_lesions):
            return None
        return tuple(self.list_lesion_loc_array[random.randint(0, self.num_lesions - 1)])

    def get_penalty_for_lesion_volume_version(self, lesion_volume_array):
        """
        :return: penalty_array.    penalty_fn[lesion_loc_array] = penalty_array
        """
        penalty_range = self.penalty_range
        if lesion_volume_array is None:
            return 0

        rescale_factor = 0.5

        power_volume = np.power(lesion_volume_array, rescale_factor)
        power_max = np.power(self.max_lesion_volume, rescale_factor) + 0.001
        power_min = np.power(self.min_lesion_volume, rescale_factor)
        penalty_array = \
            (power_volume - power_min) / (power_max - power_min) * (
                    penalty_range[1] - penalty_range[0]) + penalty_range[0]
        penalty_array = penalty_range[1] - penalty_array
        penalty_array = np.clip(penalty_array, penalty_range[0], penalty_range[1])

        return torch.FloatTensor(penalty_array)

    def __len__(self):
        return len(self.list_lesion_loc_array)


class SmoothMaskDataloader:
    def __init__(self, ct_dataset, lesion_dataset, batch_size, shuffle=False, num_workers=4, drop_last=True,
                 show=True, mode='train', num_test_replicate=1, num_lesion_applied=(0, 1), random_augment=True):
        """
        use this with:
        for batch_sample in SmoothMaskDataloader:

        batch_sample is (model_input_tensor, model_gt_output_tensor, fn_penalty_tensor, fp_penalty_tensor)
        all tensor on CPU
        model_input_tensor with shape [batch_size, 1, x, y ,z]
        model_gt_output_tensor with shape [batch_size, 2, x, y ,z], channel 0 for positive
        fn_penalty_tensor with shape [batch_size, 1, x, y ,z]
        fp_penalty_tensor with shape [batch_size, 1, x, y ,z]

        :param ct_dataset: an instance of "SmoothMaskDataset"
        :param lesion_dataset: an instance of "SimulateLesionDataset"
        :param batch_size: determine the batch_size
        :param shuffle:
        :param num_workers: number CPU during preparing the data,
        :param drop_last: cut the last few sample that less than batch size
        :param show:
        :param mode:
        :param num_test_replicate: the test set will be len(ct_dataset) * num_test_replicate
        :param num_lesion_applied: for each sample, the max number of lesion applied on this sample
        """
        self.random_augment = random_augment
        assert mode in ['train', 'test']
        if show:
            print("mode:", mode)
            print("batch_size:", batch_size, "shuffle:", shuffle, "num_workers:", num_workers, "drop_last:", drop_last,
                  "lesion_difficulty: level", lesion_dataset.difficulty_level, 'num_test_replicate', num_test_replicate)

        self.mode = mode
        self.num_test_replicate = num_test_replicate
        self.ct_dataset = []
        for index in range(len(ct_dataset)):
            self.ct_dataset.append(ct_dataset[index])

        self.lesion_dataset = lesion_dataset
        self.epoch_passed = 0
        self.lesion_difficulty_level = lesion_dataset.difficulty_level

        self.testing_sample_list = None
        self.testing_lesion_loc_array_list = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.sample_loaded = 0
        self.num_samples = len(ct_dataset)
        assert self.num_samples > batch_size

        if show:
            print("There are:", len(self.ct_dataset), "sample cubes")
            print("Lesion difficulty level:", lesion_dataset.difficulty_level,
                  "There are:", len(lesion_dataset), "lesions")

        self.show = show
        self.step_in_iter = 0

        if not type(num_lesion_applied) is tuple:
            num_lesion_applied = (0, max(num_lesion_applied))

        self.num_lesion_applied = num_lesion_applied
        print("number lesion applied:", num_lesion_applied)

    def __len__(self):
        if self.drop_last:
            return int(self.num_samples / self.batch_size)
        else:
            if self.num_samples % self.batch_size == 0:
                return int(self.num_samples / self.batch_size)
            return int(self.num_samples / self.batch_size) + 1

    def convert_sample_to_tensor(self, sample_batch, lesion_loc_array_batch):
        """
        :param sample_batch: list of sample, each sample is float32 numpy array in [4, x, y, z]
        :param lesion_loc_array_batch: list of (loc-array, volume-array), None means no lesion.
        :return: model_input_tensor, model_gt_output_tensor, weight_tensor
                  [B, 1, x, y, z]       [B, 2, x, y, z]      [B, 2, x, y, z]

                  model_gt_output_tensor, channel 0 for positive, channel 1 for negative
                  weight_tensor, channel 0 for fn_penalty, channel 1 for fp_penalty
        """
        batch_size = len(sample_batch)
        assert len(lesion_loc_array_batch) == batch_size
        model_input_tensor, model_gt_output_tensor, fn_penalty_tensor, fp_penalty_tensor = [], [], [], []
        for index in range(batch_size):
            sample_array = np.array(sample_batch[index])
            if self.random_augment:
                sample_array = random_flip_rotate_swap(sample_array, deep_copy=False)
            # initialize tensor
            input_tensor = torch.FloatTensor(sample_array[0])
            output_gt_tensor = torch.FloatTensor(sample_array[1])
            fn_tensor = torch.FloatTensor(sample_array[2])
            fp_tensor = torch.FloatTensor(sample_array[3])
            # apply lesion
            if lesion_loc_array_batch[index] is not None:
                lesion_loc_array, lesion_volume_array = lesion_loc_array_batch[index]
            else:
                lesion_loc_array, lesion_volume_array = None, None
            if lesion_loc_array is not None:
                input_tensor[lesion_loc_array] = 1
                output_gt_tensor[lesion_loc_array] = 1
                fn_tensor[lesion_loc_array] = self.lesion_dataset.get_penalty_for_lesion_volume_version(
                    lesion_volume_array)
                # the average fn penalty for voxel is around 0.15-0.5
                fp_tensor[lesion_loc_array] = 0
            model_input_tensor.append(input_tensor)
            model_gt_output_tensor.append(output_gt_tensor)
            fn_penalty_tensor.append(fn_tensor)
            fp_penalty_tensor.append(fp_tensor)

        model_input_tensor = torch.stack(model_input_tensor, dim=0)  # [B, x, y, z]
        model_input_tensor = model_input_tensor.unsqueeze(dim=1)  # [B, 1, x, y, z]

        model_gt_output_tensor = torch.stack(model_gt_output_tensor, dim=0)
        model_gt_output_tensor = torch.stack((model_gt_output_tensor, 1 - model_gt_output_tensor), dim=1)
        # [B, 2, x, y, z]

        fn_penalty_tensor = torch.stack(fn_penalty_tensor, dim=0)  # [B, x, y, z]
        fp_penalty_tensor = torch.stack(fp_penalty_tensor, dim=0)  # [B, x, y, z]
        weight_tensor = torch.stack((fn_penalty_tensor, fp_penalty_tensor), dim=1)  # [B, 2, x, y, z]

        return model_input_tensor, model_gt_output_tensor, weight_tensor

    def establish_test_dataset(self):
        print("establishing new testing dataset...")
        time_start = time.time()
        assert self.epoch_passed == 0 and self.mode == 'test'
        test_sample_list = []
        for index in range(self.num_test_replicate):
            test_sample_list = test_sample_list + self.ct_dataset[0::]

        for index in range(self.num_workers):
            receive_end, send_end = multiprocessing.Pipe(False)
            self.receive_end_list.append(receive_end)
            self.send_end_list.append(send_end)
            self.sample_batch_list.append(test_sample_list[index::self.num_workers])
        for index, sample_batch in enumerate(self.sample_batch_list):
            self.start_applying_lesion(self.send_end_list[index], sample_batch)

        self.testing_sample_list = []
        self.testing_lesion_loc_array_list = []
        for index, receive_end in enumerate(self.receive_end_list):
            self.testing_sample_list = self.testing_sample_list + self.sample_batch_list[index]
            self.testing_lesion_loc_array_list = self.testing_lesion_loc_array_list + receive_end.recv()

        self.num_samples = len(self.testing_sample_list)

        time_end = time.time()
        print("test dataset contains:", self.num_samples, 'samples')
        print("establish test dataset cost time:", time_end - time_start)

    def get_next_sample_batch_train(self):
        assert self.mode == 'train'
        count_initial = self.sample_loaded
        count_terminal = count_initial + self.batch_size

        if count_initial >= self.num_samples:
            return None

        if count_terminal >= self.num_samples:
            if self.drop_last:
                return None
            else:
                count_terminal = self.num_samples

        self.sample_loaded = count_terminal
        sample_batch = self.ct_dataset[count_initial: count_terminal]
        return sample_batch  # a list of sample

    def get_next_tensor_batch_test(self):
        assert self.mode == 'test'
        count_initial = self.sample_loaded
        count_terminal = count_initial + self.batch_size

        if count_initial >= self.num_samples:
            return None

        if count_terminal >= self.num_samples:
            if self.drop_last:
                return None
            else:
                count_terminal = self.num_samples

        self.sample_loaded = count_terminal
        sample_batch = self.testing_sample_list[count_initial: count_terminal]
        lesion_loc_array_batch = self.testing_lesion_loc_array_list[count_initial: count_terminal]
        return self.convert_sample_to_tensor(sample_batch, lesion_loc_array_batch)

    def start_applying_lesion(self, send_end, sample_batch):
        """

        start the sub process that working to calculate the lesion_loc_arrays for the given sample_batch
        """
        input_list = []
        if sample_batch is None:
            input_list = [send_end, None]
        else:
            for sample in sample_batch:
                list_lesion_loc_array = []
                for index in range(random.randint(self.num_lesion_applied[0], self.num_lesion_applied[1])):
                    list_lesion_loc_array.append(self.lesion_dataset.get_arbitrary_lesion())

                none_count = 0
                for lesion_loc_array in list_lesion_loc_array:
                    if lesion_loc_array is None:
                        none_count += 1

                if none_count == len(list_lesion_loc_array):
                    input_list.append((None, None, None))
                else:
                    input_list.append((sample[0], sample[1], list_lesion_loc_array))
            input_list = [send_end, input_list]
        # start working... process will wait until receive end extract data
        multiprocessing.Process(target=apply_lesion_parallel, args=input_list).start()

    def __iter__(self):
        print("epoch passed for this", self.mode, "dataloader:", self.epoch_passed)

        if self.shuffle:
            random.shuffle(self.ct_dataset)

        self.sample_loaded = 0
        self.step_in_iter = 0
        self.receive_end_list = []
        self.send_end_list = []
        self.sample_batch_list = []

        if self.mode == 'train':
            for index in range(self.num_workers):
                receive_end, send_end = multiprocessing.Pipe(False)
                self.receive_end_list.append(receive_end)
                self.send_end_list.append(send_end)
                self.sample_batch_list.append(self.get_next_sample_batch_train())
            for index, sample_batch in enumerate(self.sample_batch_list):
                self.start_applying_lesion(self.send_end_list[index], sample_batch)
        else:
            if self.epoch_passed == 0:
                self.establish_test_dataset()

        return self

    def __next__(self):
        if self.mode == 'train':
            # receive pre calculated lesions
            extract_index = self.step_in_iter % self.num_workers
            lesion_loc_array_batch = self.receive_end_list[extract_index].recv()
            sample_batch = self.sample_batch_list[extract_index]
            if sample_batch is None:
                self.epoch_passed += 1
                raise StopIteration()

            tensor_batch = self.convert_sample_to_tensor(sample_batch, lesion_loc_array_batch)

            # start a new process
            receive_end, send_end = multiprocessing.Pipe(False)
            self.receive_end_list[extract_index] = receive_end
            self.send_end_list[extract_index] = send_end
            self.sample_batch_list[extract_index] = self.get_next_sample_batch_train()
            self.start_applying_lesion(send_end, self.sample_batch_list[extract_index])

            self.step_in_iter += 1
            return tensor_batch
        else:
            tensor_batch = self.get_next_tensor_batch_test()
            if tensor_batch is None:
                self.epoch_passed += 1
                raise StopIteration()
            self.step_in_iter += 1
            return tensor_batch


def check_correctness_test_dataloader():
    import visualization.visualize_3d.visualize_stl as stl
    temp_lesion_dataset = SimulateLesionDataset(difficulty_level=6, penalty_range=(0.1, 1))
    temp_sample_dataset_test = SmoothMaskDataset('/data_disk/artery_vein_project/extract_blood_region/'
                                                 'training_data/sliced_sample/256_v1/CTA/stack_array_artery/',
                                                 num_workers=1, sample_interval=(0, 2), mode='test')

    temp_dataloader_test = SmoothMaskDataloader(temp_sample_dataset_test, temp_lesion_dataset, 2, shuffle=True,
                                                num_workers=8, drop_last=False, show=True, mode='test',
                                                num_test_replicate=4, num_lesion_applied=(20, 20), random_augment=False)

    for tensor_batch_test in temp_dataloader_test:
        print(temp_dataloader_test.sample_loaded)
        model_input_t, model_gt_output_t, weight_tensor = tensor_batch_test
        fn_penalty_t, fp_penalty_t = weight_tensor[:, 0], weight_tensor[:, 1]
        model_input_a = model_input_t.numpy()
        model_gt_output_a = model_gt_output_t.numpy()
        fn_penalty_a = fn_penalty_t.numpy()
        fp_penalty_a = fp_penalty_t.numpy()
        print(np.shape(model_input_a))
        print(np.shape(model_gt_output_a))
        print(np.shape(fn_penalty_a))
        print(np.shape(fp_penalty_a))
        stl.visualize_numpy_as_stl(model_input_a[0, 0])
        show_z = int(np.median(np.where(model_input_a[0, 0] > 0)[2]))
        Functions.image_show(model_input_a[0, 0, :, :, show_z])
        Functions.image_show(model_gt_output_a[0, 0, :, :, show_z])
        Functions.image_show(model_gt_output_a[0, 1, :, :, show_z])
        Functions.image_show(fn_penalty_a[0, :, :, show_z])
        Functions.image_show(fp_penalty_a[0, :, :, show_z])


def check_correctness_train_dataloader():
    import visualization.visualize_3d.visualize_stl as stl
    temp_lesion_dataset = SimulateLesionDataset(difficulty_level=6)
    temp_sample_dataset_train = SmoothMaskDataset('/data_disk/artery_vein_project/extract_blood_region/'
                                                  'training_data/sliced_sample/256_v1/CTA/stack_array_artery/',
                                                  num_workers=1, sample_interval=(0, 5), mode='train')

    temp_dataloader_train = SmoothMaskDataloader(temp_sample_dataset_train, temp_lesion_dataset, 2, shuffle=True,
                                                 num_workers=4, drop_last=False, show=True, mode='train',
                                                 num_lesion_applied=(5, 10))

    for tensor_batch_train in temp_dataloader_train:
        print(temp_dataloader_train.sample_loaded)
        model_input_t, model_gt_output_t, weight_tensor = tensor_batch_train
        fn_penalty_t, fp_penalty_t = weight_tensor[:, 0], weight_tensor[:, 1]
        model_input_a = model_input_t.numpy()
        model_gt_output_a = model_gt_output_t.numpy()
        fn_penalty_a = fn_penalty_t.numpy()
        fp_penalty_a = fp_penalty_t.numpy()
        print(np.shape(model_input_a))
        print(np.shape(model_gt_output_a))
        print(np.shape(fn_penalty_a))
        print(np.shape(fp_penalty_a))
        stl.visualize_numpy_as_stl(model_input_a[0, 0])
        show_z = int(np.median(np.where(model_input_a[0, 0] > 0)[2]))
        Functions.image_show(model_input_a[0, 0, :, :, show_z])
        Functions.image_show(model_gt_output_a[0, 0, :, :, show_z])
        Functions.image_show(model_gt_output_a[0, 1, :, :, show_z])
        Functions.image_show(fn_penalty_a[0, :, :, show_z])
        Functions.image_show(fp_penalty_a[0, :, :, show_z])


if __name__ == '__main__':
    check_correctness_test_dataloader()
