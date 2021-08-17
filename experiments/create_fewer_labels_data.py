import os
import numpy as np
from util.data import ecg_datasets2


def create_few_labels_splits(train_fraction, val_fraction, test_fraction):

    crop_size = 4500
    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling, verbose=False)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling, verbose=False)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=crop_size,
                                                     pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling, verbose=False)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling, verbose=False)



    ecg_datasets2.filter_update_classes_by_count([georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature], min_count=20)
    print("Warning! Redoing splits!")
    filename = f'train-test-splits-fewer-labels{train_fraction*100}-{val_fraction*100}-{test_fraction*100}.txt'
    ptbxl_challenge.random_train_split_with_class_count(train_fraction, val_fraction, test_fraction, filename_overwrite=filename)
    cpsc_challenge.random_train_split_with_class_count(train_fraction, val_fraction, test_fraction, filename_overwrite=filename)
    cpsc2_challenge.random_train_split_with_class_count(train_fraction, val_fraction, test_fraction, filename_overwrite=filename)
    georgia_challenge.random_train_split_with_class_count(train_fraction, val_fraction, test_fraction, filename_overwrite=filename)

    
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file(ttsfile=filename)
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file(ttsfile=filename)
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file(ttsfile=filename)
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file(ttsfile=filename)

    ecg_datasets2.filter_update_classes_by_count([nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4], 1)
    print('Classes after last update', len(ptbxl_train.classes), ptbxl_train.classes)
    print(ecg_datasets2.count_merged_classes([nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4]))


def create_few_labels_splits_like_old(train_fraction):

    crop_size = 4500
    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=crop_size,
                                                     pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)

    OLD_TRAIN_FRACTION = 0.7
    if train_fraction > OLD_TRAIN_FRACTION:
        print(f"Cant take {train_fraction} of data as train dataset, when previous was {OLD_TRAIN_FRACTION}.")
        return
    else:
        new_train_fraction = train_fraction/OLD_TRAIN_FRACTION #Between 0 and 1
        print(f"Taking {new_train_fraction} of old train split")
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file()
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file()
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file()
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file()

    #ecg_datasets2.filter_update_classes_by_count([georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature], min_count=20)
    print("Warning! Redoing splits!")
    filename = f'train-test-splits-fewer-labels{str(train_fraction)}.txt'
    print(f'saving as {filename}')
    ptbxl_new_train_split, _, _ = ptbxl_train.random_train_split_with_class_count(new_train_fraction, 0.0, 0.0, save=False)
    georgia_new_train_split, _, _ =georgia_train.random_train_split_with_class_count(new_train_fraction, 0.0, 0.0, save=False)
    cpsc_new_train_split, _, _ =cpsc_train.random_train_split_with_class_count(new_train_fraction, 0.0, 0.0, save=False)
    cpsc2_new_train_split, _, _ =cpsc2_train.random_train_split_with_class_count(new_train_fraction, 0.0, 0.0, save=False)

    ecg_datasets2.save_train_test_split(os.path.join(ptbxl_train.BASE_DIR, filename), ptbxl_new_train_split, ptbxl_val.files, t1.files)
    ecg_datasets2.save_train_test_split(os.path.join(georgia_train.BASE_DIR, filename), georgia_new_train_split, georgia_val.files, t2.files)
    ecg_datasets2.save_train_test_split(os.path.join(cpsc_train.BASE_DIR, filename), cpsc_new_train_split, cpsc_val.files, t3.files)
    ecg_datasets2.save_train_test_split(os.path.join(cpsc2_train.BASE_DIR, filename), cpsc2_new_train_split, cpsc2_val.files, t4.files)

    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file(ttsfile=filename)
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file(ttsfile=filename)
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file(ttsfile=filename)
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file(ttsfile=filename)


    ecg_datasets2.filter_update_classes_by_count([nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4], 1)
    print('Classes after last update', len(ptbxl_train.classes), ptbxl_train.classes)
    print(ecg_datasets2.count_merged_classes([nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4]))


def create_few_labels_splits_minimum_cut(min_cut=20):
    crop_size = 4500
    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=crop_size, pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=crop_size,
                                                     pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=crop_size,
                                                      pad_to_size=crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file()
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file()
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file()
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file()


    ecg_datasets2.filter_update_classes_by_count([ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4], 1)
    print('Classes after last update', len(ptbxl_train.classes), ptbxl_train.classes)

    class_counts, counted_classes = ecg_datasets2.count_merged_classes([ptbxl_train, georgia_train, cpsc_train, cpsc2_train])
    extracted_counts = class_counts
    extracted_counts[np.where(class_counts>min_cut)[0]] = min_cut
    print(extracted_counts)
    class_counts_dset = np.empty((4, len(class_counts))).astype(int)
    for i, dset in enumerate([ptbxl_train, georgia_train, cpsc_train, cpsc2_train]):
        c_count, counted_c = ecg_datasets2.count_merged_classes([dset])
        class_counts_dset[i] = np.array(c_count)
    dsets = [ptbxl_train, georgia_train, cpsc_train, cpsc2_train]
    buckets = np.zeros((4, len(class_counts))).astype(int)
    sort_idx = np.argsort(class_counts)
    for ix in sort_idx:
        #find this label in files.
        dset_order = np.argsort(class_counts_dset[:, ix])
        have = 0
        for i, dset_i in enumerate(dset_order):
            n_per_dataset = int(np.ceil((extracted_counts[ix] - have)/(len(dsets)-i)))
            cc_in_dset = class_counts_dset[dset_i, ix]
            take_n_files = min(cc_in_dset, n_per_dataset)
            buckets[dset_i, ix] = take_n_files
            have += take_n_files
    print('Found buckets', buckets)
    filename = f'train-test-splits_min_cut{min_cut}.txt'

    ptbxl_new_train_split, _, _ = ptbxl_train.random_train_split_with_class_count_mins(buckets[0], np.zeros_like(buckets[0]), np.zeros_like(buckets[0]), save=False)
    georgia_new_train_split, _, _ = georgia_train.random_train_split_with_class_count_mins(buckets[1], np.zeros_like(buckets[1]), np.zeros_like(buckets[1]), save=False)
    cpsc_new_train_split, _, _ = cpsc_train.random_train_split_with_class_count_mins(buckets[2], np.zeros_like(buckets[2]), np.zeros_like(buckets[2]), save=False)
    cpsc2_new_train_split, _, _ = cpsc2_train.random_train_split_with_class_count_mins(buckets[3], np.zeros_like(buckets[3]), np.zeros_like(buckets[3]), save=False)

    ecg_datasets2.save_train_test_split(os.path.join(ptbxl_train.BASE_DIR, filename), ptbxl_new_train_split, ptbxl_val.files, t1.files)
    ecg_datasets2.save_train_test_split(os.path.join(georgia_train.BASE_DIR, filename), georgia_new_train_split, georgia_val.files, t2.files)
    ecg_datasets2.save_train_test_split(os.path.join(cpsc_train.BASE_DIR, filename), cpsc_new_train_split, cpsc_val.files, t3.files)
    ecg_datasets2.save_train_test_split(os.path.join(cpsc2_train.BASE_DIR, filename), cpsc2_new_train_split, cpsc2_val.files, t4.files)


if __name__ == '__main__': # DO NOT CALL UNLESS YOU WANT TO OVERWRITE
    create_few_labels_splits_like_old(train_fraction=0.001)
    create_few_labels_splits_like_old(train_fraction=0.005)
    create_few_labels_splits_like_old(train_fraction=0.01)
    create_few_labels_splits_like_old(train_fraction=0.05)
    #create_few_labels_splits_like_old(train_fraction=0.1)
    # create_few_labels_splits_like_old(train_fraction=0.2)
    # create_few_labels_splits_like_old(train_fraction=0.3)
    # create_few_labels_splits_like_old(train_fraction=0.4)
    # create_few_labels_splits_like_old(train_fraction=0.5)
    # create_few_labels_splits_like_old(train_fraction=0.6)
    # create_few_labels_splits_minimum_cut(min_cut=200)
    # create_few_labels_splits_minimum_cut(min_cut=150)
    # create_few_labels_splits_minimum_cut(min_cut=100)
    # create_few_labels_splits_minimum_cut(min_cut=50)
    # create_few_labels_splits_minimum_cut(min_cut=25)
    #create_few_labels_splits_minimum_cut(min_cut=3)