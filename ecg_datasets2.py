from random import shuffle

import torch
import os
import h5py
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch._six import container_abcs, string_classes, int_classes

from external.helper_code import find_challenge_files, load_recording


class ECGDataset(torch.utils.data.IterableDataset):
    def __init__(self, BASE_DIR, window_size, n_windows, files=None, channels=None, use_labels=False, preload_windows=0): #TODO: option to load into ram
        super(ECGDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.n_windows = n_windows
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()
        self.channels = channels
        self.total_length = 1 #Trying a weird approach (calculated in __iter__)
        self.use_labels = use_labels
        self.preload_windows = preload_windows

    def __iter__(self):
        #multiple workers?
        file_index = 0
        self.total_length = 0
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                offset = np.random.randint(0, self.window_size) #Random offset
                if 'data' in f.keys():
                    data = f.get('data') #Make sure this exists in your dataset
                if 'labels' in f.keys():
                    labels = f.get('labels')
                if data is None:
                    print(self.current_file, "'data' table is None.")
                else:
                    if not self.use_labels:
                        while (index+1)*self.window_size*self.n_windows+offset <= len(data):
                            preloaded = data[index*self.window_size*self.n_windows+offset: (index+1+self.preload_windows)*self.window_size*self.n_windows+offset].copy() #maybe copy
                            preload_index = 0
                            while (preload_index+1)*self.window_size*self.n_windows+offset <= len(preloaded):
                                if self.channels:
                                    yield np.swapaxes(preloaded[preload_index*self.window_size*self.n_windows+offset: (preload_index+1)*self.window_size*self.n_windows+offset, self.channels].reshape((self.n_windows, self.window_size, -1)), 1, 2) #TODO: pad data in between with 0?
                                else:
                                    yield np.swapaxes(preloaded[preload_index * self.window_size * self.n_windows + offset: (preload_index+1) * self.window_size * self.n_windows + offset, :].reshape((self.n_windows, self.window_size, -1)), 1, 2)
                                preload_index += 1
                                index += 1

                    else:
                        while (index+1)*self.window_size*self.n_windows+offset <= len(data):
                            preloaded = data[index*self.window_size*self.n_windows+offset: (index+1+self.preload_windows)*self.window_size*self.n_windows+offset].copy()
                            preload_index = 0
                            while (preload_index+1)*self.window_size*self.n_windows+offset <= len(preloaded):
                                if self.channels:
                                    yield np.swapaxes(preloaded[preload_index*self.window_size*self.n_windows+offset: (preload_index+1)*self.window_size*self.n_windows+offset, self.channels].reshape((self.n_windows, self.window_size, -1)), 1, 2) #TODO: pad data in between with 0?
                                else:
                                    yield np.swapaxes(preloaded[preload_index * self.window_size * self.n_windows + offset: (preload_index+1) * self.window_size * self.n_windows + offset, :].reshape((self.n_windows, self.window_size, -1)), 1, 2)
                                preload_index += 1
                                index += 1
                            del preloaded
                self.total_length += len(data)
            file_index += 1

    def __len__(self):
        return self.total_length

    def search_files(self):
        record_files = []
        file_endings = ('.h5')
        for root, dirs, files in os.walk(self.BASE_DIR):
            for file in files:
                if file.endswith(file_endings):
                    record_files.append(os.path.join(root, file))
        print(len(record_files), 'record files found in ', self.BASE_DIR)
        return record_files

    def print_file_attributes(self):
        with h5py.File(self.files[0], 'r') as f: #only look at first file
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)

class ECGMultipleDatasets(torch.utils.data.IterableDataset):
    def __init__(self, torch_datasets):
        super(ECGMultipleDatasets).__init__()
        self.torch_datasets = torch_datasets

    def __iter__(self): #desired shape: (batches, windows
        #TODO: how long? ALso: custom batching?
        dataset_iterators = [iter(t) for t in self.torch_datasets]
        available = list(range(len(self.torch_datasets)))
        available2 = list(range(len(self.torch_datasets)))
        while available:
            for i in available:
                try:
                    yield next(dataset_iterators[i])
                except StopIteration:
                    available2.remove(i)
            available = available2.copy()

    def __len__(self):
        return sum(map(len, self.torch_datasets))  # Whatever

    def print_file_attributes(self):
        for dataset in self.torch_datasets:
            with h5py.File(dataset.files[0], 'r') as f:  # only look at first file
                print('Keys contained in dataset:', f.keys())
                for k in f.keys():
                    data = f.get(k)
                    print('Data with key [%s] has shape:' % k, data.shape)


class ECGDatasetBatching(ECGDataset): #TODO: make preload available?
    def __init__(self, BASE_DIR, window_size, n_windows, window_gap=0, files=None, channels=None, use_labels=False, use_random_offset=False, preload_windows=0, batch_size=1, data_max_len=10000): #TODO: option to load into ram
        super(ECGDatasetBatching, self).__init__(BASE_DIR, window_size, n_windows, files, channels, use_labels, preload_windows)
        self.batch_size = batch_size
        self.window_gap = window_gap
        self.use_random_offsets = use_random_offset
        self.data_max_len = data_max_len

    def __iter__(self):
        #multiple workers?
        file_index = 0
        self.total_length = 0
        available_files = list(range(len(self.files)))
        if available_files:
            strike = 0
            selected_files = np.random.choice(available_files, self.batch_size, replace=False)
            opened_files = [h5py.File(self.files[f_ind], 'r') for f_ind in selected_files] #open all selected files
            data_indices = np.zeros(self.batch_size, dtype=int)
            half_hearthrate = 800//2
            data_offsets = np.random.randint(0, self.data_max_len-self.window_size*self.n_windows, self.batch_size, dtype=int)
            for sel in selected_files:
                available_files.remove(sel) #remove selected for future draw
            while opened_files:
                for i, f in enumerate(opened_files):
                    data_index = data_indices[i]
                    offset = data_offsets[i]
                    if 'data' in f.keys():
                        data = f.get('data')  # Make sure this exists in your dataset
                    if 'multilabel' in f.keys():
                        labels = f.get('multilabel')
                    elif 'label' in f.keys():
                        labels = f.get('label')
                    if (data_index + 1) * self.window_size * self.n_windows + offset <= len(data): #+ self.window_gap
                        #print('using:', i, opened_files[i])
                        if not self.use_labels:
                            if self.channels:
                                yield np.swapaxes(data[data_index*self.window_size*self.n_windows + offset: (data_index+1)*self.window_size*self.n_windows+offset, self.channels].reshape((self.n_windows, self.window_size, -1)), 1, 2) #TODO: pad data in between with 0?
                            else:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (data_index+1) * self.window_size * self.n_windows + offset, :].reshape((self.n_windows, self.window_size, -1)), 1, 2)
                        else:
                            if self.channels:
                                yield np.swapaxes(data[data_index*self.window_size*self.n_windows+offset: (data_index+1)*self.window_size*self.n_windows+offset, self.channels].reshape((self.n_windows, self.window_size, -1)), 1, 2), np.array((data_index + 2) * self.window_size * self.n_windows + offset > len(data)), labels[:]
                            else:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (data_index+1) * self.window_size * self.n_windows + offset, :].reshape((self.n_windows, self.window_size, -1)), 1, 2), np.array((data_index + 2) * self.window_size * self.n_windows + offset > len(data)), labels[:]
                        data_indices[i] += 1
                        self.total_length += self.window_size * self.n_windows
                    #Replace used up files:
                    if (data_index + 2) * self.window_size * self.n_windows + offset > len(data): # Remove this from opened. TODO: what about unused data?
                        if available_files:
                            #print('opening new for', i)
                            opened_files[i] = h5py.File(self.files[available_files.pop()], 'r')
                            f.close()  # closing here
                        else:
                            strike += 1
                        data_indices[i] = 0
                        data_offsets[i] = np.random.randint(0, self.window_size - 1)
                        if strike >= self.batch_size:  # Recycle until all files have been fully used
                            return

class ECGLabelDataset(torch.utils.data.IterableDataset):
    def __init__(self, BASE_DIR, window_size, n_windows, files=None):
        super(ECGLabelDataset).__init__()
        self.BASE_DIR = BASE_DIR
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.window_size = window_size
        self.n_windows = n_windows
        self.print_file_attributes()

    def __iter__(self):
        #multiple workers?
        file_index = 0
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                offset = np.random.randint(self.window_size) #Use a random offset
                data = f.get('data')
                labels = f.get('labels') #Uses data and labels group in .h5py file. make sure those exist
                while (index+1)*self.window_size*self.n_windows <= len(data):
                    #tmp_data = data[index*self.window_size*self.n_windows:(index+1)*self.window_size*self.n_windows]
                    yield np.swapaxes(data[index*self.window_size*self.n_windows+offset : (index+1)*self.window_size*self.n_windows+offset].reshape((self.n_windows, self.window_size, -1)), 1, 2),\
                          np.swapaxes(labels[index*self.window_size*self.n_windows+offset : (index+1)*self.window_size*self.n_windows+offset].reshape((self.n_windows, self.window_size, -1)), 1, 2)#labels[self.n_windows:(index+1)*self.window_size*self.n_windows] #TODO: Try returning as list
                    index += 1
            file_index += 1


    def search_files(self):
        record_files = []
        file_endings = ('.h5')
        for root, dirs, files in os.walk(self.BASE_DIR):
            for file in files:
                if file.endswith(file_endings):
                    record_files.append(os.path.join(root, file))
        print(len(record_files), 'record files found in ', self.BASE_DIR)
        return record_files

    def print_file_attributes(self):
        print('Printing attributes for dataset at dir:', self.BASE_DIR)
        with h5py.File(self.files[0], 'r') as f: #only look at first file
            print('looking at file', f)
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)

            labels = f.get('labels')  # Uses data and labels group in .h5py file. make sure those exist
            print(np.min(labels), np.max(labels))


class ECGDatasetBaseline(torch.utils.data.IterableDataset):

    def __init__(self, BASE_DIR, window_size, files=None):
        super(ECGDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()

    def __iter__(self):
        file_index = 0
        shuffle(self.files)
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                data = f.get('data')
                labels = f.get('label')
                offset = np.random.randint(len(data) - self.window_size)  # Random offset
                if not data is None and not labels is None:
                    yield data[offset:self.window_size + offset, :], labels[:]
            file_index += 1

    def __len__(self):
        return 1  # Whatever

    def search_files(self):
        record_files = []
        file_endings = ('.h5')
        for root, dirs, files in os.walk(self.BASE_DIR):
            for file in files:
                if file.endswith(file_endings):
                    record_files.append(os.path.join(root, file))
        print(len(record_files), 'record files found in ', self.BASE_DIR)
        return record_files

    def print_file_attributes(self):
        with h5py.File(self.files[0], 'r') as f:  # only look at first file
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)

class ECGDatasetBaselineMulti(torch.utils.data.IterableDataset):

    def __init__(self, BASE_DIR, window_size, files=None):
        super(ECGDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()

    def __iter__(self):
        file_index = 0
        shuffle(self.files)
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                data = f.get('data')
                labels = f.get('multilabel')
                offset = np.random.randint(len(data) - self.window_size)  # Random offset
                if not data is None and not labels is None:
                    yield data[offset:self.window_size + offset, :], labels[:]
            file_index += 1

    def __len__(self):
        return 1  # Whatever

    def search_files(self):
        record_files = []
        file_endings = ('.h5')
        for root, dirs, files in os.walk(self.BASE_DIR):
            for file in files:
                if file.endswith(file_endings):
                    record_files.append(os.path.join(root, file))
        print(len(record_files), 'record files found in ', self.BASE_DIR)
        return record_files

    def print_file_attributes(self):
        with h5py.File(self.files[0], 'r') as f:  # only look at first file
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)

class ECGChallengeDataset(torch.utils.data.IterableDataset):
    def __init__(self, BASE_DIR, window_size, n_windows, window_gap=0, files=None, channels=None, use_labels=False,
                 use_random_offset=False, preload_windows=0, batch_size=1):
        super(ECGDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.n_windows = n_windows
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()
        self.channels = channels
        self.total_length = 1 #Trying a weird approach (calculated in __iter__)
        self.use_labels = use_labels
        self.use_random_offsets = use_random_offset
        self.batch_size = batch_size
        self.window_gap = window_gap


    def __iter__(self):
        self.total_length = 0
        available_files = list(range(len(self.files)))
        if available_files:
            strike = 0
            selected_files = np.random.choice(available_files, self.batch_size, replace=False)
            opened_file_data = [self._read_recording_file(self.files[f_ind]) for f_ind in selected_files]  # open all selected files
            if self.use_labels:
                opened_header_data = []
            data_indices = np.zeros(self.batch_size, dtype=int)
            data_offsets = np.random.randint(0, self.window_size, self.batch_size, dtype=int)
            for sel in selected_files:
                available_files.remove(sel)  # remove selected for future draw
            while opened_file_data:
                for i, f in enumerate(opened_file_data):
                    data_index = data_indices[i]
                    offset = data_offsets[i]
                    data = None #TODO: implement
                    if self.use_labels:
                        pass
                    if (data_index + 1) * self.window_size * self.n_windows + offset <= len(data):  # + self.window_gap
                        # print('using:', i, opened_files[i])
                        if not self.use_labels:
                            if self.channels:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (
                                                                                                                            data_index + 1) * self.window_size * self.n_windows + offset,
                                                  self.channels].reshape((self.n_windows, self.window_size, -1)), 1,
                                                  2)  # TODO: pad data in between with 0?
                            else:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (
                                                                                                                            data_index + 1) * self.window_size * self.n_windows + offset,
                                                  :].reshape((self.n_windows, self.window_size, -1)), 1, 2)
                        else:
                            if self.channels:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (
                                                                                                                            data_index + 1) * self.window_size * self.n_windows + offset,
                                                  self.channels].reshape((self.n_windows, self.window_size, -1)), 1,
                                                  2), np.array(
                                    (data_index + 2) * self.window_size * self.n_windows + offset > len(data)), labels[
                                                                                                                :]
                            else:
                                yield np.swapaxes(data[data_index * self.window_size * self.n_windows + offset: (
                                                                                                                            data_index + 1) * self.window_size * self.n_windows + offset,
                                                  :].reshape((self.n_windows, self.window_size, -1)), 1, 2), np.array(
                                    (data_index + 2) * self.window_size * self.n_windows + offset > len(data)), labels[
                                                                                                                :]
                        data_indices[i] += 1
                        self.total_length += self.window_size * self.n_windows
                    # Replace used up files:
                    if (data_index + 2) * self.window_size * self.n_windows + offset > len(
                            data):  # Remove this from opened. TODO: what about unused data?
                        if available_files:
                            # print('opening new for', i)
                            opened_file_data[i] = h5py.File(self.files[available_files.pop()], 'r')
                            f.close()  # closing here
                        else:
                            strike += 1
                        data_indices[i] = 0
                        data_offsets[i] = np.random.randint(0, self.window_size - 1)
                        if strike >= self.batch_size:  # Recycle until all files have been fully used
                            return

    def _read_recording_file(self, path_without_ext):
        fp = path_without_ext + '.mat'
        return load_recording(fp, key='val')

    def __len__(self):
        return self.total_length

    def search_files(self):
        headers, records = find_challenge_files(self.BASE_DIR)



def collate_fn(batch): #https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

