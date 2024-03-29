import os
from random import shuffle

import h5py
import numpy as np
import torch


class ECGDataset(torch.utils.data.IterableDataset):
    def __init__(self, BASE_DIR, window_size, n_windows, files=None):
        super(ECGDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.n_windows = n_windows
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()

    def __iter__(self):
        # multiple workers?
        file_index = 0
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                offset = np.random.randint(0, self.window_size)  # Random offset
                data = f.get('data')  # Make sure this exists in your dataset
                if not data is None:
                    while (index + 1) * self.window_size * self.n_windows + offset <= len(data):
                        yield np.swapaxes(data[index * self.window_size * self.n_windows + offset: (
                                                                                                               index + 1) * self.window_size * self.n_windows + offset,
                                          0:2].reshape((self.n_windows, self.window_size, -1)), 1, 2)
                        index += self.n_windows
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
        # multiple workers?
        file_index = 0
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                offset = np.random.randint(self.window_size)  # Use a random offset
                data = f.get('data')
                labels = f.get('labels')  # Uses data and labels group in .h5py file. make sure those exist
                while (index + 1) * self.window_size * self.n_windows <= len(data):
                    # tmp_data = data[index*self.window_size*self.n_windows:(index+1)*self.window_size*self.n_windows]
                    yield np.swapaxes(data[index * self.window_size * self.n_windows + offset: (
                                                                                                           index + 1) * self.window_size * self.n_windows + offset].reshape(
                        (self.n_windows, self.window_size, -1)), 1, 2), \
                          np.swapaxes(labels[index * self.window_size * self.n_windows + offset: (
                                                                                                             index + 1) * self.window_size * self.n_windows + offset].reshape(
                              (self.n_windows, self.window_size, -1)), 1,
                                      2)  # labels[self.n_windows:(index+1)*self.window_size*self.n_windows] #TODO: Try returning as list
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
        with h5py.File(self.files[0], 'r') as f:  # only look at first file
            print('looking at file', f)
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)

            labels = f.get('labels')  # Uses data and labels group in .h5py file. make sure those exist
            print(np.min(labels), np.max(labels))


class ECGDatasetMultiple(torch.utils.data.IterableDataset):
    def __init__(self, BASE_DIRECTORIES, window_size, n_windows, files=None, selected_channels=None):
        super(ECGDatasetMultiple).__init__()
        self.BASE_DIRECTORIES = BASE_DIRECTORIES
        self.n_windows = n_windows
        self.window_size = window_size
        if files:
            self.files = files
        else:
            self.files = self.search_files()
        self.print_file_attributes()

    def __iter__(self):
        # multiple workers?
        file_index = 0
        while file_index < len(self.files):
            self.current_file = self.files[file_index]
            with h5py.File(self.current_file, 'r') as f:
                index = 0
                offset = np.random.randint(0, self.window_size)  # Random offset
                data = f.get('data')  # Make sure this exists in your dataset
                while (index + 1) * self.window_size * self.n_windows + offset <= len(data):
                    yield np.swapaxes(data[index * self.window_size * self.n_windows + offset: (
                                                                                                       index + 1) * self.window_size * self.n_windows + offset,
                                      0:2].reshape(
                        (self.n_windows, self.window_size, -1)), 1,
                        2)  # TODO: MAKE chosen channel index selectable for each dataset
                    index += self.n_windows
            file_index += 1

    def __len__(self):
        return 1  # Whatever

    def search_files(self):
        record_files = []
        file_endings = ('.h5')
        for base_dir in self.BASE_DIRECTORIES:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(file_endings):
                        record_files.append(os.path.join(root, file))
        print(len(record_files), 'record files found in ', self.BASE_DIRECTORIES)
        return record_files

    def print_file_attributes(self):
        with h5py.File(self.files[0], 'r') as f:  # only look at first file
            print('Keys contained in dataset:', f.keys())
            for k in f.keys():
                data = f.get(k)
                print('Data with key [%s] has shape:' % k, data.shape)


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
