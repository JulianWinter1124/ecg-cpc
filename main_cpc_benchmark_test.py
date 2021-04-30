import argparse
import datetime
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ChainDataset

from util.metrics import training_metrics
from external import helper_code
from util.data import ecg_datasets2, ptbxl_data
from util.full_class_name import fullname
from util.store_models import load_model_checkpoint, load_model_architecture, extract_model_files_from_dir

def main(args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu_device)
    print(args.out_path)
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out_path, 'params.txt'), 'w') as cfg:
        cfg.write(str(args))
    # georgia = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/georgia/WFDB', window_size=4500, pad_to_size=4500, use_labels=True)
    # cpsc_train = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc_train', window_size=4500, pad_to_size=4500, use_labels=True)
    # cpsc = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc', window_size=4500, pad_to_size=4500, use_labels=True)
    # ptbxl = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/ptbxl/WFDB', window_size=4500, pad_to_size=4500, use_labels=True)

    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                                  window_size=args.crop_size, pad_to_size=args.crop_size,
                                                                  return_labels=True, return_filename=True,
                                                                  normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                               window_size=args.crop_size, pad_to_size=args.crop_size,
                                                               return_labels=True, return_filename=True,
                                                               normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge',
                                                                window_size=args.crop_size, pad_to_size=args.crop_size,
                                                                return_labels=True, return_filename=True,
                                                                normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge',
                                                                window_size=args.crop_size, pad_to_size=args.crop_size,
                                                                return_labels=True, return_filename=True,
                                                                normalize_fn=ecg_datasets2.normalize_feature_scaling)



    a1, b1, ptbxl_test = ptbxl_challenge.generate_datasets_from_split_file()
    a2, b2, georgia_test = georgia_challenge.generate_datasets_from_split_file()
    a3, b3, cpsc_test = cpsc_challenge.generate_datasets_from_split_file()
    a4, b4, cpsc2_test = cpsc2_challenge.generate_datasets_from_split_file()


    classes = ecg_datasets2.filter_update_classes_by_count([a1, b1, ptbxl_test, a2, b2, georgia_test, a3, b3, cpsc_test, a4, b4, cpsc2_test], 1) #Set classes if specified in split files (filter out classes with no occurence)
    print(classes)
    print(classes=={'10370003': 0, '11157007': 1, '111975006': 2, '164861001': 3, '164865005': 4, '164867002': 5, '164873001': 6, '164884008': 7, '164889003': 8, '164890007': 9, '164909002': 10, '164917005': 11, '164930006': 12, '164931005': 13, '164934002': 14, '164947007': 15, '164951009': 16, '17338001': 17, '195042002': 18, '195080001': 19, '195126007': 20, '233917008': 21, '251120003': 22, '251146004': 23, '251180001': 24, '251200008': 25, '251266004': 26, '251268003': 27, '253352002': 28, '266249003': 29, '270492004': 30, '27885002': 31, '284470004': 32, '39732003': 33, '413844008': 34, '425419005': 35, '425623009': 36, '426177001': 37, '426434006': 38, '426627000': 39, '426761007': 40, '426783006': 41, '427084000': 42, '427172004': 43, '427393009': 44, '428417006': 45, '428750005': 46, '429622005': 47, '445118002': 48, '445211001': 49, '446358003': 50, '446813000': 51, '47665007': 52, '54329005': 53, '55930002': 54, '59118001': 55, '59931005': 56, '63593006': 57, '6374002': 58, '67198005': 59, '67741000119109': 60, '698252002': 61, '713422000': 62, '713426002': 63, '713427006': 64, '74390002': 65, '89792004': 66}
)
    test_dataset_challenge = ChainDataset([ptbxl_test, georgia_test, cpsc_test, cpsc2_test])
    #all_dataset_challenge = ChainDataset[ptbxl_challenge, georgia_challenge, cpsc_challenge, cpsc2_challenge]
    model_folders = [
        #'models/01_03_21-14'
        #'models/04_03_21-14',
        'models/30_04_21-18'

    ]
    #infer class from model-arch file
    models = []
    for mfolder in model_folders:
        model_files = extract_model_files_from_dir(mfolder)
        for mfile in model_files:
            fm_fs, cp_fs = mfile
            fm_f = fm_fs[0]
            cp_f = sorted(cp_fs)[-1]
            model = load_model_architecture(fm_f)
            model, _, epoch = load_model_checkpoint(cp_f, model, optimizer=None)
            models.append(model)
    loaders = [
        DataLoader(test_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn),
        # DataLoader(all_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
        #            collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score
        training_metrics.micro_avg_precision_score,
        training_metrics.micro_avg_recall_score,
    ]
    for model_i, model in enumerate(models):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name+str(model_i))
        print("Evaluating {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))
        with open(os.path.join(output_path, 'model_arch.txt'), 'w') as f:
            print(fullname(model), file=f)
            print(model, file=f)
        model.cuda()
        # init optimizer
        optimizer = Adam(model.parameters(), lr=3e-4)
        metrics = defaultdict(lambda: defaultdict(list))
        pred_dataframe = pd.DataFrame(columns=classes)
        pred_dataframe.index.name = 'filename'
        label_dataframe = pd.DataFrame(columns=classes)
        label_dataframe.index.name = 'filename'
        for epoch in range(1, 2):
            starttime = time.time()  # train
            for loader_i, loader in enumerate(loaders):
                for dataset_tuple in loader:
                    data, labels, filenames = dataset_tuple
                    data = data.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                    pred = model(data, y=None)  # makes model return prediction instead of loss
                    pred = pred.detach().cpu()
                    labels = labels.cpu()
                    labels_numpy = parse_tensor_to_numpy_or_scalar(labels)
                    pred_numpy = parse_tensor_to_numpy_or_scalar(pred)
                    pred_dataframe = pred_dataframe.append(pd.DataFrame(pred_numpy, columns=classes, index=filenames))
                    label_dataframe = label_dataframe.append(pd.DataFrame(labels_numpy, columns=classes, index=filenames))
                    helper_code.save_challenge_predictions(output_path, filenames, classes=classes, scores=pred_numpy, labels=labels_numpy)
                    with torch.no_grad():
                        for i, fn in enumerate(metric_functions):
                            metrics[epoch]['acc_' + str(i)].append(
                                parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                    if args.dry_run:
                        break
                    del data, pred, labels
                csv_pred_name = "model-{}-dataloader-{}-output.csv".format(model_name, loader_i)
                csv_label_name = "labels-dataloader-{}.csv".format(loader_i)
                print("\tFinished dataset {}. Progress: {}/{}".format(loader_i, loader_i + 1, len(loaders)))
                print("\tSaving prediction and label to csv.")
                pred_dataframe.to_csv(os.path.join(output_path, csv_pred_name))
                label_dataframe.to_csv(os.path.join(output_path, csv_label_name))
                print("\tSaved files {} and {}".format(csv_pred_name, csv_label_name))
                torch.cuda.empty_cache()

            elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
            metrics[epoch]['elapsed_time'].append(elapsed_time)
            print("Done. Elapsed time: {}".format(elapsed_time))
            if args.dry_run:
                break

        pickle_name = "model-{}-test.pickle".format(model_name)
        # Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i + 1, len(models)))
        del model  # delete and free
        torch.cuda.empty_cache()

def save_dict_to_csv_file(filepath, data, column_names=None):
    with open(filepath) as f:
        if not column_names is None:
            f.write(','.join(column_names)+'\n')
        for dc in data:
            f.write(','.join(dc)+'\n')


def parse_tensor_to_numpy_or_scalar(input_tensor):
    if type(input_tensor) == torch.Tensor:
        arr = input_tensor.detach().cpu().numpy() if input_tensor.is_cuda else input_tensor.numpy()
        if arr.size == 1:
            return arr.item()
        return arr
    return input_tensor

if __name__ == "__main__":
    import sys

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Contrastive Predictive Coding')
    # datapath
    # Other params
    parser.add_argument('--saved_model', type=str,
                        help='Model path to load weights from. Has to be given for downstream mode.')

    parser.add_argument('--epochs', type=int, help='The number of Epochs to train', default=100)

    parser.add_argument('--seed', type=int, help='The seed used', default=None)

    parser.add_argument('--forward_mode', help="The forward mode to be used.", default='context',
                        type=str)  # , choices=['context, latents, all']

    parser.add_argument('--out_path', help="The output directory for losses and models",
                        default='models/' + str(datetime.datetime.now().strftime("%d_%m_%y-%H")), type=str)

    parser.add_argument('--forward_classes', type=int, default=52,
                        help="The number of possible output classes (only relevant for downstream)")

    parser.add_argument('--warmup_steps', type=int, default=0, help="The number of warmup steps")

    parser.add_argument('--batch_size', type=int, default=24, help="The batch size")

    parser.add_argument('--latent_size', type=int, default=128,
                        help="The size of the latent encoding for one window")

    parser.add_argument('--timesteps_in', type=int, default=6,
                        help="The number of windows being used to form a context for prediction")

    parser.add_argument('--timesteps_out', type=int, default=6,
                        help="The number of windows being predicted from the context (cpc task exclusive)")

    parser.add_argument('--crop_size', type=int, default=4500,
                        help="The size of the data that it is cropped to. If data is smaller than this number, data gets padded with zeros")

    parser.add_argument('--channels', type=int, default=12,
                        help="The number of channels the data will have")  # TODO: auto detect

    parser.add_argument('--window_size', type=int, default=512,
                        help="The number of datapoints per channel per window")

    parser.add_argument('--hidden_size', type=int, default=512,
                        help="The size of the cell state/context used for predicting future latents or solving downstream tasks")

    parser.add_argument('--dry_run', dest='dry_run', action='store_true',
                        help="Only run minimal samples to test all models functionality")
    parser.set_defaults(dry_run=False)

    parser.add_argument("--gpu_device", type=int, default=0)

    args = parser.parse_args()
    main(args)
