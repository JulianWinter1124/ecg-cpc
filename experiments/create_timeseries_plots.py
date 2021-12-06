import json
import os
import numpy as np
from torch.utils.data import DataLoader, ChainDataset

from util.data import ecg_datasets2
from util.visualize.timeseries_to_image_converter import timeseries_to_image


def create_timeseries_image():
    crop_size = 4500
    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                                  window_size=crop_size, pad_to_size=crop_size,
                                                                  return_labels=True, return_filename=True,
                                                                  normalize_fn=ecg_datasets2.normalize_minmax_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                               window_size=crop_size, pad_to_size=crop_size,
                                                               return_labels=True, return_filename=True,
                                                               normalize_fn=ecg_datasets2.normalize_minmax_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge',
                                                                window_size=crop_size,
                                                                pad_to_size=crop_size, return_labels=True,
                                                                return_filename=True,
                                                                normalize_fn=ecg_datasets2.normalize_minmax_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge',
                                                                window_size=crop_size,
                                                                pad_to_size=crop_size, return_labels=True,
                                                                return_filename=True,
                                                                normalize_fn=ecg_datasets2.normalize_minmax_scaling)
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file()
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file()
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file()
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file()
    ecg_datasets2.filter_update_classes_by_count(
        [ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val,
         t4], 1)
    counts_all, counted_classes_all = ecg_datasets2.count_merged_classes(
        [ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val,
         t4])
    print(counted_classes_all)
    with open('/experiments/snomed_data.json', 'r') as f:
        snomed_data = json.load(f)
    inverted_named_counted_classes = {v: snomed_data[k]['pt']['term'] for k, v in counted_classes_all.items()}
    dl = DataLoader(ChainDataset([t1, t2, t3, t4]), batch_size=1)
    for data, labels, filenames in dl:
        data = data.float().cpu()
        labels = labels.cpu().numpy()
        print(np.nonzero(labels[0])[0])
        fname = os.path.basename(filenames[0])
        ground_truth = [inverted_named_counted_classes[i] for i in np.nonzero(labels[0])[0]]
        timeseries_to_image(data, grad=None, pred_classes=None, ground_truth=[ground_truth], save=True, show=True,
                            downsample_factor=1,
                            filename=f'/home/julian/Downloads/Github/contrastive-predictive-coding/images/data_visualization/{fname}')


if __name__ == '__main__':  # DO NOT CALL UNLESS YOU WANT TO OVERWRITE
    create_timeseries_image()
