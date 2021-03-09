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
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ChainDataset

import accuracy_metrics
from external import helper_code
from util.data import ecg_datasets2
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
    window_size = 4500
    georgia = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=window_size, pad_to_size=window_size, return_labels=True, return_filename=True)
    cpsc_train = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=window_size, pad_to_size=window_size, return_labels=True, return_filename=True)
    cpsc = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=window_size,
                                                     pad_to_size=window_size, return_labels=True, return_filename=True)
    ptbxl = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=window_size,
                                                      pad_to_size=window_size, return_labels=True, return_filename=True)

    georgia.merge_and_update_classes([georgia, cpsc, ptbxl, cpsc_train])
    classes = georgia.classes
    train_dataset_challenge = ChainDataset([georgia, cpsc_train, cpsc])
    all_dataset_challenge = ChainDataset([georgia, cpsc_train, cpsc, ptbxl])
    val_dataset_challenge = ChainDataset([ptbxl])

    model_folders = [
        #'models/01_03_21-14'
        'models/04_03_21-14'
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
        DataLoader(val_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn),
        # DataLoader(all_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
        #            collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score
        accuracy_metrics.micro_avg_precision_score,
        accuracy_metrics.micro_avg_recall_score,
    ]
    for model_i, model in enumerate(models):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name)
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

    parser.add_argument('--channels', type=int, default=12,
                        help="The number of channels the data will have")  # TODO: auto detect

    parser.add_argument('--window_length', type=int, default=512,
                        help="The number of datapoints per channel per window")

    parser.add_argument('--hidden_size', type=int, default=512,
                        help="The size of the cell state/context used for predicting future latents or solving downstream tasks")

    parser.add_argument('--dry_run', dest='dry_run', action='store_true',
                        help="Only run minimal samples to test all models functionality")
    parser.set_defaults(dry_run=False)

    parser.add_argument("--gpu_device", type=int, default=0)

    args = parser.parse_args()
    main(args)
