#!/usr/bin/env python
# coding: utf-8

# In[11]:

import os
from collections import defaultdict

import h5py
import numpy as np
import wfdb
# In[18]:
from sklearn.preprocessing import normalize, OneHotEncoder


# ### Read MIT format .dat ecg data files and .hea headers


class ECGData():
    def __init__(self, base_directory='/home/julian/Datasets/ptb-diagnostic-ecg-database-1.0.0/'):
        self.BASE_DIR = base_directory
        self.label_mappings = {}
        self.default_key_function_dict = defaultdict(lambda y: (lambda x: True, 2))
        #self.default_key_function_dict['age'] = lambda x: int(x) > 50 if x.isnumeric() else False, 2  # age $TODO: more onehots
        #self.default_key_function_dict['smoker'] = lambda x: x == 'yes', 2
        self.default_key_function_dict['reason for admission'] = lambda x: self.filter_comment('reason for admission', x), 10
        self.label_encoder = None
        self.pca = None

    def search_files(self, file_endings=None):
        root = self.BASE_DIR
        if file_endings is None:
            file_endings = ['.dat', '.hea', '.xyz']
        record_files = []
        with open(os.path.join(root, 'RECORDS')) as recs:
            record_files = recs.read().splitlines()
        print(len(record_files), 'record files found in ', root)
        return record_files

    def read_signal(self, record_path, physical=True):
        record = wfdb.rdrecord(record_path, physical=physical)
        if physical:
            data = record.p_signal
        else:
            data = record.d_signal
        return data

    def read_header(self, record_path):
        record = wfdb.rdheader(record_path)
        return record.comments

    def parse_comment_dict(self, wfdb_comment):
        comment_map = {}
        for c in wfdb_comment:
            e = c.lower().split(':')
            comment_map[e[0]] = e[1].strip()
        return comment_map

    def filter_comment(self, key, comment_string):
        c = comment_string
        if key == 'reason for admission':
            if 'cardiomyopathy' in c or 'heart failure' in c:
                return 'cardiomyopathy'
            elif 'n/a' in c or 'palpitation' in c:
                return 'miscellaneous'
            elif 'angina' in c:
                return 'angina'
        else:
            print('given key not found', key)
        return comment_string

    def onehot_comment(self, wfdb_comment, terms: list = [], key_function_dict:dict = None):
        values = []
        column_names = []
        if terms:
            terms_encoded = False
            for i, term in enumerate(terms):
                column_names.append(term)
                for c in wfdb_comment:
                    if term in c:
                        terms_encoded = True
                        break
            values.append(terms_encoded)
        comment_dict = self.parse_comment_dict(wfdb_comment)
        for key, (func, n) in key_function_dict.items():
            if key in comment_dict:
                value = func(comment_dict[key])
                values.append(value)
                column_names.append(key)
        print('values', values)
        return self.label_encoder.transform([values]).toarray(), column_names

    def init_label_encoder(self, files, terms, key_function_dict:dict = None):
        self.label_encoder = OneHotEncoder()
        X = []
        for i, file in enumerate(files):
            absolute = os.path.join(self.BASE_DIR, file)
            wfdb_comment = self.read_header(absolute)
            comment_dict = self.parse_comment_dict(wfdb_comment)
            temp = [i % 2 == 0]*len(terms) #Show possible values
            for key, (func, n) in key_function_dict.items():
                if key in comment_dict:
                    value = func(comment_dict[key])
                    temp.append(value)
            X.append(temp)
        print(X)

        self.label_encoder.fit(X)
        print(self.label_encoder.categories_)



    def read_all_files(self, record_path_list):
        file_data = []
        for f in record_path_list:
            d = self.read_signal(os.path.join(self.BASE_DIR, f))
            file_data.append(d)
        return file_data

    def partition_data(self, data, window_size=3000, overlap=0.5, store_in_array=True, align_right=True,
                       verbose=True):  # maybe allow non float overlap too
        samples, channels = data.shape
        if samples < window_size:
            print('too few samples (%d) to support window size of %d' % (samples, window_size))
            return None
        if verbose: print('Input data has shape:', data.shape)
        shift = window_size * (1-overlap)
        offset = int(samples % shift)
        if align_right:
            used_data = data[offset:]
        else:
            used_data = data[:-offset]
        samples, _ = used_data.shape
        if verbose:
            print('The window of size %d will be shifted by %f. The total data used is %d' % (window_size, shift, samples))
        partitioned = np.empty((int(samples / shift) - 1, window_size, channels))
        if verbose: print('The partitioned data now has shape:', partitioned.shape)
        for i in range(len(partitioned)):
            index = int(i * shift)
            partitioned[i, :, :] = used_data[index:index + window_size, :]
        return partitioned

    def generate_context_task(self, partitioned_data, N, observations=5, predictions=3, verbose=True,
                              shuffle_all=False):  # maybe use shuffle?
        # generate N-1 negative samples and 1 positive sample:
        windows, samples, channels = partitioned_data.shape
        positive_samples_x = []
        positive_samples_y = []
        negative_samples_x = []
        negative_samples_y = []
        for i in range(0, windows - observations - predictions):
            i_cur = i + observations
            positive_samples_x += [partitioned_data[i:i_cur, :, :]]
            positive_samples_y += [partitioned_data[i_cur:i_cur + predictions]]
            possible_choices = list(range(i)) + list(range(i_cur + predictions, windows))
            for _ in range(N - 1):
                choices = np.random.choice(possible_choices, predictions,
                                           replace=False)  # advanced use p param for different probabilities
                negative_samples_x += [partitioned_data[i:i_cur, :, :]]
                negative_samples_y += [partitioned_data[choices, :, :]]

        return positive_samples_x, positive_samples_y, negative_samples_x, negative_samples_y

    def convert_dat_to_h5_partitioned(self, storage_path, dat_file_paths, window_size=3000, overlap=0.5, align_right=True, overwrite=True, verbose=True):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        for f in dat_file_paths:
            data = self.read_signal(os.path.join(self.BASE_DIR, f))
            print(f)
            partitioned = self.partition_data(data, window_size, overlap, True, align_right, verbose)
            target = os.path.join(storage_path, f.replace('/', '-') +'.h5')
            with h5py.File(target, 'w') as wf:
                wf['data'] = partitioned
                wf.flush()
                if verbose: print(target, 'file created and written. %d windows saved.' % (len(partitioned)))

    def convert_dat_to_h5(self, storage_path, dat_file_paths, channels=None, normalize_data=False, pca_components=0, use_labels=False, use_header=False, terms = None, key_function_dict=None, verbose=True):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        for f in dat_file_paths:
            absolute = os.path.join(self.BASE_DIR, f)
            if not channels:
                data = self.read_signal(absolute)
            else:
                data = self.read_signal(absolute)[:, channels]
            if normalize_data:
                data = normalize(data, norm='l2')
            if pca_components > 0:
                print('performing pca with:', data.shape)
                cov = np.cov(data.T)
                eig, ev = np.linalg.eigh(cov)
                evecs = ev[::-1][:, 0:pca_components] #order is ascending, so descend, then take first two
                data = np.dot(data, evecs)
                print('output shape pca:', data.shape)
            target = os.path.join(storage_path, f.replace('/', '-') +'.h5')
            with h5py.File(target, 'w') as wf:
                wf['data'] = data #TODO: Make this a parameter (drop last 3 channels)
                wf.flush()
                if use_header:
                    comments = self.read_header(absolute)
                    onehot, column_names = self.onehot_comment(comments, terms, key_function_dict)
                    wf['label'] = onehot
                    wf['label'].attrs.create('names', column_names)
                    wf.flush()
                if verbose: print(target, 'file created and written')


    def convert_dat_to_h5_context_task(self, storage_path, dat_file_paths, window_size=3000, overlap=0.5, time_in=5, time_out=3, align_right=True, verbose=True):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        for f in dat_file_paths:
            data = self.read_signal(os.path.join(self.BASE_DIR, f))
            print(f)
            partitioned = self.partition_data(data, window_size, overlap, True, align_right, verbose)
            p_x, p_y, n_x, n_y = self.generate_context_task(partitioned, 100, time_in, time_out, verbose, False)
            
            target = os.path.join(storage_path, f.replace('/', '-') +'.h5')
            with h5py.File(target, 'w') as wf:
                wf['windows'] = partitioned
                wf.flush()
                if verbose: print(target, 'file created and written. %d windows saved.' % (len(partitioned)))


