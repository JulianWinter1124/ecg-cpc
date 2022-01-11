import os
from random import shuffle

import torch
from torch.utils.data import IterableDataset

import numpy as np

from external import helper_code
from util.utility import timestamp
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format


class ECGChallengeDatasetBaseline(IterableDataset):
    def __init__(self, BASE_DIR, window_size, pad_to_size=None, files=None, channels=None, return_labels=False,
                 return_filename=False, classes=None, normalize_fn=None, verbose=False, randomize_order=True):
        super(IterableDataset).__init__()
        self.BASE_DIR = BASE_DIR
        self.window_size = window_size
        self.pad_to_size = pad_to_size or window_size
        self.files = files or self.search_files()
        self.classes = classes or helper_code.get_classes(self.files)
        if verbose:
            self.print_file_attributes()
        self.channels = channels
        self.total_length = 1  # Trying a weird approach (calculated in __iter__)
        self.return_labels = return_labels
        self.return_filename = return_filename
        self.normalize_fn = normalize_fn if not normalize_fn is None else lambda x: x
        self.randomize_order = randomize_order
        self.loaded_data = {}
        self.loaded_labels = {}

    def generate_datasets_from_split_file(self, ttsfile='train-test-splits.txt'):
        splits = load_train_test_split(os.path.join(self.BASE_DIR, ttsfile))
        return tuple(ECGChallengeDatasetBaseline(self.BASE_DIR, self.window_size, self.pad_to_size, files=s,
                                                 channels=self.channels, return_labels=self.return_labels,
                                                 return_filename=self.return_filename, classes=self.classes,
                                                 normalize_fn=self.normalize_fn, randomize_order=self.randomize_order)
                     for s in splits)

    def generate_datasets_from_split_list(self, trainf, valf, testf):
        splits = [trainf, valf, testf]
        return tuple(ECGChallengeDatasetBaseline(self.BASE_DIR, self.window_size, self.pad_to_size, files=s,
                                                 channels=self.channels, return_labels=self.return_labels,
                                                 return_filename=self.return_filename, classes=self.classes,
                                                 normalize_fn=self.normalize_fn, randomize_order=self.randomize_order)
                     for s in splits)

    def preload(self, preload_frac=0.):
        self.loaded_data = {}
        self.loaded_labels = {}
        for f in self.files[0:int(len(self.files)*preload_frac)]:
            d = torch.from_numpy(self.normalize_fn(self._read_recording_file(f)))
            if len(d) - self.window_size < 0: #pad
                d = torch.nn.ReflectionPad1d((0, max(0, self.pad_to_size - min(self.window_size, len(d)))))(d.T.unsqueeze(0)).squeeze(0).T #WTF is this pytorch
            self.loaded_data[f] = d
            #del d
            self.loaded_labels[f] = torch.from_numpy(self._read_header_labels(f))


    def __iter__(self):
        file_index = 0
        if self.randomize_order:
            shuffle(self.files)
        while file_index < len(self.files):
            current_file = self.files[file_index]
            if current_file in self.loaded_data:
                data = self.loaded_data[current_file]
            else:
                data = torch.Tensor(self.normalize_fn(self._read_recording_file(current_file)))
            if self.channels:
                data = data[:, self.channels]
            if self.return_labels:
                if current_file in self.loaded_labels:
                    labels = self.loaded_labels[current_file]
                else:
                    labels = torch.Tensor(self._read_header_labels(current_file))
            if len(data) - self.window_size >= 0:
                if self.randomize_order and (len(data) - self.window_size) > 0:
                    offset = np.random.randint(len(data) - self.window_size)  # Random offset
                else:
                    offset = 0
                data = data[offset:self.window_size + offset]
            else:
                offset = 0
                data = torch.nn.ReflectionPad1d((0, max(0, self.pad_to_size - min(self.window_size, len(data)))))(data.T.unsqueeze(0)).squeeze(0).T
            if not any([self.return_filename, self.return_labels]):
                yield data
            else:
                yield [data] + ([labels] if self.return_labels else [None]) + (
                    [current_file] if self.return_filename else [])

            file_index += 1

    def _read_recording_file(self, path_without_ext):
        fp = path_without_ext + '.mat'
        return helper_code.load_recording(fp, key='val').transpose()

    def _read_header_file(self, path_without_ext):
        fp = path_without_ext + '.hea'
        return helper_code.load_header(fp)

    def _read_header_labels(self, path_without_ext, onerror_class='426783006'):
        header = self._read_header_file(path_without_ext)
        return helper_code.encode_header_labels(header, self.classes, onerror_class)

    def __len__(self):
        return self.total_length

    def search_files(self):
        headers, records = helper_code.find_challenge_files(self.BASE_DIR)
        print(len(records), 'record files found in ', self.BASE_DIR)
        return list(map(lambda x: os.path.splitext(x)[0], records))  # remove extension

    def random_train_split(self, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1, save=True,
                           save_path_overwrite=None, filename_overwrite=None):
        assert train_fraction + val_fraction + test_fraction <= 1
        N = len(self.files)
        shuffle(self.files)
        train_slice = slice(0, int(train_fraction * N))
        val_slice = slice(train_slice.stop, train_slice.stop + int(val_fraction * N))
        test_slice = slice(val_slice.stop, val_slice.stop + int(test_fraction * N))
        if save:
            p = save_path_overwrite or self.BASE_DIR
            fname = filename_overwrite or 'train-test-splits.txt'
            save_train_test_split(os.path.join(p, fname), self.files[train_slice], self.files[val_slice],
                                  self.files[test_slice])
        return self.files[train_slice], self.files[val_slice], self.files[test_slice]

    def random_train_split_with_class_count(self, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1, save=True,
                                            save_path_overwrite=None, filename_overwrite=None):
        assert train_fraction + val_fraction + test_fraction <= 1
        class_buckets = [[] for x in range(len(self.classes))]  # Creates classes buckets
        for i, f in enumerate(self.files):
            label = self._read_header_labels(f)
            label_idx = np.argwhere(label).flatten()
            for l in label_idx:  # can return more than one (multi-label)
                class_buckets[l].append(f)  # Put this file into class l bucket
        train_files, val_files, test_files = [], [], []
        sorted_idx = list(sorted(range(len(class_buckets)),
                                 key=lambda x: len(class_buckets[x])))  # make index sorted by class count low->high
        print(sorted_idx)
        while len(sorted_idx) > 0:
            idx = sorted_idx[0]
            shuffle(class_buckets[idx])
            b = class_buckets[idx]
            c_N = len(b)
            train_slice = slice(0, int(train_fraction * c_N))
            val_slice = slice(train_slice.stop, train_slice.stop + int(val_fraction * c_N))
            test_slice = slice(val_slice.stop, val_slice.stop + int(test_fraction * c_N))
            train_files += b[train_slice]
            val_files += b[val_slice]
            test_files += b[test_slice]
            used_set = set(b[train_slice] + b[val_slice] + b[test_slice])
            for j in sorted_idx[1:]:
                class_buckets[j] = [x for x in class_buckets[j] if
                                    x not in used_set]  # REMOVE THIS FILE FROM ALL OTHER BUCKETS
            sorted_idx = list(
                sorted(sorted_idx[1:], key=lambda x: len(class_buckets[x])))  # sort again (removal may change order)

        train_files = list(set(train_files))
        val_files = list(set(val_files))
        test_files = list(set(test_files))
        if save:
            p = save_path_overwrite or self.BASE_DIR
            fname = filename_overwrite or 'train-test-splits.txt'
            save_train_test_split(os.path.join(p, fname), train_files, val_files, test_files)
        return train_files, val_files, test_files

    def random_train_split_with_class_count_mins(self, train_counts, val_counts, test_counts, save=True,
                                                 save_path_overwrite=None, filename_overwrite=None):
        class_buckets = [[] for x in range(len(self.classes))]  # Creates classes buckets
        for i, f in enumerate(self.files):
            label = self._read_header_labels(f)
            label_idx = np.argwhere(label).flatten()
            for l in label_idx:  # can return more than one (multi-label)
                class_buckets[l].append(f)  # Put this file into class l bucket
        file_splits = [[], [], []]
        for i, split_counts in enumerate([test_counts, val_counts, train_counts]):
            sorted_class_idxs = np.argsort(split_counts)
            sorted_class_idxs = sorted_class_idxs[np.argwhere(split_counts[sorted_class_idxs] > 0)].ravel()
            while (len(sorted_class_idxs) > 0):  # TODO:add bad counter
                for class_idx in sorted_class_idxs:
                    if len(class_buckets[class_idx]) == 0:
                        split_counts[class_idx] = 0
                        continue
                    shuffle(class_buckets[class_idx])
                    file_splits[i].append(class_buckets[class_idx].pop())
                    split_counts[class_idx] -= 1

                sorted_class_idxs = np.argsort(split_counts)
                sorted_class_idxs = sorted_class_idxs[np.argwhere(split_counts[sorted_class_idxs] > 0)].ravel()

        train_files = list(set(file_splits[2]))
        val_files = list(set(file_splits[1]))
        test_files = list(set(file_splits[0]))
        if save:
            p = save_path_overwrite or self.BASE_DIR
            fname = filename_overwrite or 'train-test-splits.txt'
            save_train_test_split(os.path.join(p, fname), train_files, val_files, test_files)
        return train_files, val_files, test_files

    def count_classes(self):
        counts = np.zeros(len(self.classes), dtype=int)
        for i, f in enumerate(self.files):
            labels = self._read_header_labels(f).astype(float)
            counts += labels != 0.0  # Count where label isnt 0
        return counts

    def train_split_with_function(self, file_mapping_function, save_path=None):
        splits = [[], [], []]
        for f in self.files:
            i = file_mapping_function(f)
            splits[i].append(f)
        if not save_path is None:
            save_train_test_split(os.path.join(save_path, 'train-test-splits.txt'), splits[0], splits[1], splits[2])
        return splits[0], splits[1], splits[2]

    def print_file_attributes(self):
        f = self.files[0]
        print('Information for file', f)
        data = self._read_recording_file(f)
        print('Data has shape:', data.shape)
        header = self._read_header_file(f)
        print('Header is:', header, end='###############\n')
        print('Classes found in data folder:', self.classes)
        labels = self._read_header_labels(f)
        print('Labels have shape', labels.shape)

    def merge_and_update_classes(self, datasets):
        all_classes = set()
        for d in datasets:
            all_classes = all_classes | set(d.classes.keys())
        all_classes = dict(zip(sorted(all_classes), range(len(all_classes))))
        for d in datasets:
            d.classes = all_classes
        print('Labels for datasets set to:', all_classes)

    def remove_unknown_label_files(self):
        for f in self.files[:]:
            if self._read_header_labels(f, onerror_class=None) is None:
                print('removed', f)
                self.files.remove(f)


def filter_update_classes_by_count(datasets, min_count, add_unknown=False):
    counts, all_classes = count_merged_classes(datasets)
    filtered_classes = set()
    for k, v in all_classes.items():
        if counts[v] >= min_count:
            filtered_classes.add(k)
    if add_unknown:
        filtered_classes.add('-1')
    filtered_classes = dict(zip(sorted(filtered_classes), range(len(filtered_classes))))
    for d in datasets:
        d.classes = filtered_classes
        d.remove_unknown_label_files()
    return filtered_classes


def count_merged_classes(datasets):
    all_classes = set()
    for d in datasets:
        all_classes = all_classes | set(d.classes.keys())
    all_classes = sorted(all_classes)
    all_classes = dict(zip(all_classes, range(len(all_classes))))
    counts = np.zeros(len(all_classes), dtype=int)
    for d in datasets:
        temp_classes = d.classes.copy()  # set back later
        d.classes = all_classes
        counts += d.count_classes()
        d.classes = temp_classes  # set back
    return counts, all_classes


def load_train_test_split(tts_file_path: str):
    splits = [[], [], []]
    with open(tts_file_path, 'r') as f:
        line_count = 0
        for line in f:
            if not line.strip().startswith('#') or line == '\n':  # Comment line or empty line
                splits[line_count] = [f.strip() for f in line.split(',')]
                line_count += 1
            if line_count >= 3:
                break
    return splits[0], splits[1], splits[2]


def save_train_test_split(tts_file: str, trainf=[], valf=[], testf=[]):
    if os.path.isfile(tts_file):  # make a backup just in case
        sf = os.path.split(tts_file)
        print(os.path.join(sf[0], timestamp.string_timestamp_minutes()) + sf[1])
        os.rename(tts_file, os.path.join(sf[0], timestamp.string_timestamp_minutes()) + sf[1])

    with open(tts_file, 'w') as f:
        f.write('#train files\n')
        f.write(",".join(trainf) + '\n')
        f.write('#val files\n')
        f.write(",".join(valf) + '\n')
        f.write('#test files\n')
        f.write(",".join(testf))


def collate_fn(batch):  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
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
    elif issubclass(type(elem), int):
        return torch.tensor(batch)
    elif issubclass(type(elem), str):
        return batch
    elif issubclass(type(elem), dict):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif issubclass(type(elem), list):
        # check to make sure that the elements in batch have consistent size
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]


def normalize_minmax_scaling(data, low: int = 0, high: int = 1, axis=0):
    data = data - data.min(axis=axis, keepdims=True)
    data = data / np.maximum(data.max(axis=axis, keepdims=True), 1e-12)
    data = data * (high-low)
    data = data + low
    return data

def normalize_minmax_scaling_different(data, low: int = 0, high: int = 1, axis=1):
    data = data - data.min(axis=axis, keepdims=True)
    data = data / np.maximum(data.max(axis=axis, keepdims=True), 1e-12)
    data = data * (high-low)
    data = data + low
    return data

def normalize_std_scaling(data, axis=0):
    data = data - data.mean(axis=axis, keepdims=True)
    data = data / np.maximum(data.std(axis=axis, keepdims=True), 1e-12)
    return data

