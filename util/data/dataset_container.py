from util.data import ecg_datasets2
import os


class DatasetContainer():
    def __init__(self, data_dirs: list, window_size=4500, pad_to_size=4500, return_labels=True,
                 normalize_fn=ecg_datasets2.normalize_feature_scaling):
        self.databases = {}
        for dir in data_dirs:
            name = os.path.split(dir)[1]
            count = 0
            while name + str(count) in self.databases: count += 1
            name = name + str(count)
            self.databases[name] = ecg_datasets2.ECGChallengeDatasetBaseline(dir, window_size, pad_to_size,
                                                                             return_labels, normalize_fn)

        self.georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline(
            '/media/julian/data/data/ECG/georgia_challenge/',
            window_size=4500, pad_to_size=4500, return_labels=True,
            normalize_fn=ecg_datasets2.normalize_feature_scaling)
        self.cpsc_challenge_train = ecg_datasets2.ECGChallengeDatasetBaseline(
            '/media/julian/data/data/ECG/cps2018_challenge/',
            window_size=4500, pad_to_size=4500, return_labels=True,
            normalize_fn=ecg_datasets2.normalize_feature_scaling)
        self.cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge',
                                                                        window_size=4500,
                                                                        pad_to_size=4500, return_labels=True,
                                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
        self.ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge',
                                                                         window_size=4500,
                                                                         pad_to_size=4500, return_labels=True,
                                                                         normalize_fn=ecg_datasets2.normalize_feature_scaling)
        self.nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database',
                                                                window_size=4500,
                                                                pad_to_size=4500, return_labels=True,
                                                                normalize_fn=ecg_datasets2.normalize_feature_scaling)
