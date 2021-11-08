# from util.ecg_data import ECGData
import os
from random import shuffle

from util.data.ptbxl_data import PTBXLData


def ptb_clean():
    ecg = ECGData(base_directory='/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/')
    out_path = '/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized-pca/'
    train_val_test_ratio = 0.7, 0.2, 0.1
    assert sum(train_val_test_ratio) <= 1.0
    h5_files = ecg.search_files()
    shuffle(h5_files)
    n = len(h5_files)
    train_len, val_len = int(n * train_val_test_ratio[0]), int(n * train_val_test_ratio[1])
    terms = []  # empty for now? Requires multi target loss
    ecg.init_label_encoder(h5_files, terms, ecg.default_key_function_dict)
    ecg.convert_dat_to_h5(os.path.join(out_path, 'train'),
                          h5_files[0:train_len], normalize_data=True, use_header=True, terms=terms,
                          key_function_dict=ecg.default_key_function_dict, pca_components=2, channels=slice(0, 12))
    ecg.convert_dat_to_h5(os.path.join(out_path, 'val'),
                          h5_files[train_len:train_len + val_len], normalize_data=True, use_header=True, terms=terms,
                          key_function_dict=ecg.default_key_function_dict, pca_components=2, channels=slice(0, 12))
    ecg.convert_dat_to_h5(os.path.join(out_path, 'test'),
                          h5_files[train_len + val_len:], normalize_data=True, use_header=True, terms=terms,
                          key_function_dict=ecg.default_key_function_dict, pca_components=2, channels=slice(0, 12))
    print(ecg.label_mappings)


def ptbxl_clean():
    ecg = PTBXLData(
        base_directory='/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/')
    out_path = '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/'
    record_files_relative = ecg.search_files()
    ecg.init_label_encoder()
    ecg.init_multilabel_encoder()
    train, val, test = ecg.train_test_split(record_files_relative)
    print("final split: train %d; validation %d; test %d" % (len(train), len(val), len(test)))
    ecg.convert_dat_to_h5(os.path.join(out_path, 'train'),
                          train, normalize_data=True, use_labels=True, pca_components=0, channels=slice(0, 12))
    ecg.convert_dat_to_h5(os.path.join(out_path, 'val'),
                          val, normalize_data=True, use_labels=True, pca_components=0, channels=slice(0, 12))
    ecg.convert_dat_to_h5(os.path.join(out_path, 'test'),
                          test, normalize_data=True, use_labels=True, pca_components=0, channels=slice(0, 12))


if __name__ == '__main__':
    # ptb_clean()
    ptbxl_clean()
