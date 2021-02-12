import ast
import glob
import math
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import wfdb
# In[18]:
from sklearn.preprocessing import normalize, OneHotEncoder


# ### Read MIT format .dat ecg data files and .hea headers


class PTBXLData():
    def __init__(self, base_directory='/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'):
        self.BASE_DIR = base_directory
        self.label_encoder = None
        self.pca = None

    def search_files(self, file_endings=None, relative=True):
        root = self.BASE_DIR
        if file_endings is None:
            file_endings = ['.dat', '.hea', '.xyz']
        record_files = []
        for fe in file_endings:
            record_files += list(glob.glob(os.path.join(root, '*'+fe)))
        if relative:
            record_files = [os.path.basename(f) for f in record_files]
        record_files = list(set([os.path.splitext(f)[0] for f in record_files]))
        print(record_files)
        print(len(record_files), 'record files found in ', root, 'matching', file_endings)
        return record_files

    def read_signal(self, record_path, physical=True):
        print(record_path)
        record = wfdb.rdrecord(record_path, physical=physical)#
        if physical:
            data = record.p_signal
        else:
            data = record.d_signal
        return data

    def read_header(self, record_path):
        record = wfdb.rdheader(record_path)
        return record.comments

    def convert_dat_to_h5(self, storage_path, dat_file_paths, channels=None, normalize_data=False, pca_components=0, use_labels=False, verbose=True):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        if use_labels:
            ptbxl_database_dataframe = self.read_ptbxl_database()
            spc_codes_dataframe = self.read_ptbxl_scp_statements()
        for f in dat_file_paths:
            absolute = os.path.join(self.BASE_DIR, f)
            print(absolute)
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
            fail = False
            with h5py.File(target, 'w') as wf:
                wf['data'] = data #TODO: Make this a parameter (drop last 3 channels)
                if use_labels:
                    temp = self.read_label(ptbxl_database_dataframe, spc_codes_dataframe, f)
                    if len(temp) > 0:
                        labels, likelihoods = zip(*temp[0:1]) #TODO: more than 1 label
                        onehot = self.label_encoder.transform([labels]).toarray()
                        wf['label'] = onehot
                        wf['label'].attrs.create('names', [np.array(o, dtype=str) for o in self.label_encoder.categories_])
                        wf['multilabel'] = self.file_codes_onehot[f]
                        wf['multilabel'].attrs.create('names', [np.array(o, dtype=str) for o in self.code_list])
                    else:
                        print(f, "has no diagnostic labels")
                        fail = True
                wf.flush()
            if fail:
                os.remove(target)
                print('############# REMOVED', target)
            else:
                if verbose: print(target, 'file created and written')

    def init_label_encoder(self):
        self.label_encoder = OneHotEncoder()
        df = self.read_ptbxl_scp_statements()
        X = [[v] for v in df['diagnostic_class'].dropna().unique()]
        self.label_encoder.fit(X)
        print(self.label_encoder.categories_)

    def init_multilabel_encoder(self):
        dbdf = self.read_ptbxl_database()
        all_labels = defaultdict(set)
        # Collect all possible labels
        codes = dbdf['scp_codes']
        for row in codes:
            lbl_dict = ast.literal_eval(row)
            for k, v in lbl_dict.items():
                all_labels[k].update({v})

        # Filter out labels that have only 0 probablity
        all_labels = {k: v for k, v in all_labels.items() if max(v) > 0.0}

        # Build a dict with filename as key and scp code as values
        filenames = dbdf[['filename_hr', 'scp_codes']]
        file_codes = defaultdict(dict)
        for i, (f, c) in filenames.iterrows():
            lbl_dict = ast.literal_eval(c)
            for k, v in lbl_dict.items():
                if k in all_labels and v > 0.0:  # First check not necessary
                    file_codes[f][k] = v / 100.0

        code_indices = dict(zip(all_labels.keys(), range(len(all_labels.keys()))))
        self.file_codes_onehot = dict()
        for k, v in file_codes.items():
            hot_prob = np.zeros(len(code_indices))
            for ck, cv in v.items():
                hot_prob[code_indices[ck]] = cv
            self.file_codes_onehot[os.path.basename(k)] = hot_prob

        self.code_list = [a[0] for a in sorted(code_indices.items(), key=lambda x: x[1])]
        print(self.code_list)
        print(self.file_codes_onehot)

    def read_all_files(self, record_path_list):
        file_data = []
        for f in record_path_list:
            d = self.read_signal(os.path.join(self.BASE_DIR, f))
            file_data.append(d)
        return file_data

    def read_ptbxl_database(self):
        csvfile = os.path.join(self.BASE_DIR, 'ptbxl_database.csv')
        dataframe = pd.read_csv(csvfile)
        return dataframe

    def read_ptbxl_scp_statements(self):
        csvfile = os.path.join(self.BASE_DIR, 'scp_statements.csv')
        dataframe = pd.read_csv(csvfile)
        return dataframe

    def train_test_split(self, record_files_relative):
        df = self.read_ptbxl_database()
        selection = df[['strat_fold', 'filename_hr', 'filename_lr']]
        print('Files found in DB:', len(selection))
        train, val, test = [], [], []
        rfrset = set(record_files_relative)
        for i, [fold, fhr, flr] in selection.iterrows():
            fhr = os.path.basename(fhr)
            flr = os.path.basename(flr)
            if fhr in rfrset:
                fhr = fhr
                if fold <= 8: #https://physionet.org/content/ptb-xl/1.0.1/ #Cross-validation Folds
                    train.append(fhr)
                elif fold == 9:
                    val.append(fhr)
                elif fold == 10:
                    test.append(fhr)
            if flr in rfrset:
                flr = flr
                if fold <= 8: #https://physionet.org/content/ptb-xl/1.0.1/ #Cross-validation Folds
                    train.append(flr)
                elif fold == 9:
                    val.append(flr)
                elif fold == 10:
                    test.append(flr)

        return train, val, test

    def read_label(self, ptbxl_database_dataframe, spc_codes_dataframe, filename, likelihood_threshold=0.0):
        df = ptbxl_database_dataframe
        scp_df = spc_codes_dataframe
        rf = filename
        temp = rf #If you need to match files put it here

        row = df.loc[(df['filename_hr'].str.contains(temp)) | (df['filename_lr'].str.contains(temp))]
        if len(row) < 1:
            print(rf, 'not found in dataframe')
            return
        code = row['scp_codes'].values[0]

        labels = [(c.split(':')[0].replace('{', '').replace("'", '').strip(), c.split(':')[1].replace('}', '').strip()) for c in code.split(',')]
        print(labels)
        diagnostic_classes = []
        for l, p in labels:
            if float(p) > likelihood_threshold:
                scp_row = scp_df.loc[(scp_df.iloc[:, 0] == l) | (scp_df['diagnostic_subclass'] == l)]
                if len(scp_row) < 1:
                    print(l, 'not found in scp_statements')
                    break
                v = scp_row['diagnostic_class'].values[0]
                if not(type(v) == float and math.isnan(v)):
                    diagnostic_classes.append((v, p))
        return sorted(diagnostic_classes, key=lambda x: x[1], reverse=True)