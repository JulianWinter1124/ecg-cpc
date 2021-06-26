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
from torchviz import make_dot

from util import store_models
from util.metrics import training_metrics
from external import helper_code
from util.data import ecg_datasets2, ptbxl_data
from util.full_class_name import fullname
from util.store_models import load_model_checkpoint, load_model_architecture, extract_model_files_from_dir

from architectures_various import explain_network
from util.visualize.timeseries_to_image_converter import timeseries_to_image


def main(args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu_device)
    print(f'Device set to : {torch.cuda.current_device()}. Selected was {args.gpu_device}')
    torch.cuda.manual_seed(args.seed)
    print(f'Seed set to : {args.seed}.')
    print(f'Model outputpath: {args.out_path}')
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
    classes_by_index = {v: k for k, v in classes.items()}
    train_dataset_challenge = ChainDataset([a1,a2,a3,a4])
    val_dataset_challenge = ChainDataset([b1,b2,b3,b4])
    test_dataset_challenge = ChainDataset([ptbxl_test, georgia_test, cpsc_test, cpsc2_test])
    #all_dataset_challenge = ChainDataset[ptbxl_challenge, georgia_challenge, cpsc_challenge, cpsc2_challenge]
    model_folders = [
        'models/21_05_21-11-train|bl_cnn_v0+bl_cnn_v0_1+bl_cnn_v0_2+bl_cnn_v0_3+bl_cnn_v1+bl_cnn_v14+bl_cnn_v2+bl_cnn_v3+bl_cnn_v4+bl_cnn_v5+bl_cnn_v6+bl_cnn_v8+bl_cnn_v9/architectures_baseline_challenge.baseline_cnn_v0.BaselineNet0|dte:120'
    ]
    #infer class from model-arch file
    model_dicts = []
    for train_folder in model_folders:
        model_files = extract_model_files_from_dir(train_folder)
        for mfile in model_files:
            fm_fs, cp_fs, root = mfile
            fm_f = fm_fs[0]
            cp_f = sorted(cp_fs)[-1]
            model = load_model_architecture(fm_f)
            if model is None:
                continue
            model, _, epoch = load_model_checkpoint(cp_f, model, optimizer=None, device_id=f'cuda:{args.gpu_device}')
            print(f'Found architecturefile {os.path.basename(fm_f)}, checkpointfile {os.path.basename(cp_f)} in folder {root}. Apppending model for testing.')
            explain_model = explain_network.ExplainLabel(model)
            model_dicts.append({'model':explain_model, 'model_folder':root})
    if len(model_dicts) == 0:
        print(f"Could not find any models in {model_folders}.")
    loaders = [
        DataLoader(test_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn),
        DataLoader(val_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn),
        # DataLoader(train_dataset_challenge, batch_size=args.batch_size, drop_last=False, num_workers=1,
        #            collate_fn=ecg_datasets2.collate_fn), #Train usually not required

    ]
    SHOW = True
    SAVE = False
    for model_i, model_dict in enumerate(model_dicts):
        model = model_dict['model']
        model.eval()
        model.cpu()
        # first = True
        # init optimizer
        optimizer = Adam(model.parameters(), lr=3e-4)
        for epoch in range(1, 2):
            for loader_i, loader in enumerate(loaders):
                for dataset_tuple in loader:
                    data, labels, filenames = dataset_tuple
                    data = data.float().cpu()
                    optimizer.zero_grad()
                    pred, grad = model(data, y=None)
                    grad = grad.detach().cpu()
                    if len(pred.shape) == 1: #hack for squeezed batch dimension
                        pred = pred.unsqueeze(0)
                    pred = pred.detach().cpu()
                    print(pred.shape)
                    top3 = np.argsort(pred)[:, -3:]
                    #TODO: Iterate over output != 0 and calc grad respectively
                    pred_prob = []
                    for i, t3 in enumerate(top3):
                        pred_prob.append(list(zip([classes_by_index[t] for t in t3], pred[i, t3])))
                    not0 = [[i for i, e in enumerate(a) if e != 0.0] for a in labels]

                    ground_truth = []
                    for i, t3 in enumerate(not0):
                        ground_truth.append(list(zip([classes_by_index[t] for t in t3], labels[i, t3])))
                    img = timeseries_to_image(data.cpu(), grad, downsample_factor=5, convert_to_rgb=False, pred_classes=pred_prob, ground_truth=ground_truth, filename='images/ptbxl/gradient/ptbxl_timeseries_', show=SHOW, save=SAVE)
                    if args.dry_run:
                        break
                    del data, pred, labels
                print("\tFinished dataset {}. Progress: {}/{}".format(loader_i, loader_i + 1, len(loaders)))
                torch.cuda.empty_cache()
            if args.dry_run:
                break
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

    parser.add_argument('--seed', type=int, help='The seed used', default=0)

    parser.add_argument('--forward_mode', help="The forward mode to be used.", default='context',
                        type=str)  # , choices=['context, latents, all']

    parser.add_argument('--out_path', help="The output directory for losses and models",
                        default='models/' + str(datetime.datetime.now().strftime("%d_%m_%y-%H")) + '-test', type=str)

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
