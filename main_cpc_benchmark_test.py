import argparse
import datetime
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.data import ecg_datasets2
from util.full_class_name import fullname
from util.store_models import load_model_checkpoint, load_model_architecture, extract_model_files_from_dir

def main(args):
    np.random.seed(args.seed)

    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out_path, 'params.txt'), 'w') as cfg:
        cfg.write(args)
    train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
        '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        window_size=9500)

    trainloader = DataLoader(train_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1,
                             collate_fn=ecg_datasets2.collate_fn)

    val_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
        '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
        # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
        window_size=9500)
    valloader = DataLoader(val_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1,
                           collate_fn=ecg_datasets2.collate_fn)

    model_folders = [
        'models/25_02_21-13'
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
        trainloader,
        valloader
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score

    ]
    for model_i, model in enumerate(models):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name)
        print("Evaluating {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))

        model.cuda()
        # init optimizer
        optimizer = Adam(model.parameters(), lr=3e-4)
        metrics = defaultdict(lambda: defaultdict(list))
        for epoch in range(1, args.epochs + 1):
            starttime = time.time()  # train
            for loader_i, loader in enumerate(loaders):
                for dataset_tuple in loader:
                    data, labels = dataset_tuple
                    data = data.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                    pred = model(data, y=None)  # makes model return prediction instead of loss

                    # saving metrics
                    metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                    with torch.no_grad():
                        for i, fn in enumerate(metric_functions):
                            metrics[epoch]['acc_' + str(i)].append(
                                parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                    if args.dry_run:
                        break
                    del data, pred, labels
                print("\tFinished training dataset {}. Progress: {}/{}".format(loader_i, loader_i + 1,
                                                                               len(loaders)))

                torch.cuda.empty_cache()

            elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
            metrics[epoch]['elapsed_time'].append(elapsed_time)
            print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f} Elapsed time: {}".format(
                epoch, args.epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']),
                elapsed_time))
            if args.dry_run:
                break
        pickle_name = "model-{}-epochs-{}.pickle".format(model_name, args.epochs)
        # Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i + 1, len(models)))
        del model  # delete and free
        torch.cuda.empty_cache()



def parse_tensor_to_numpy_or_scalar(input_tensor):
    arr = input_tensor.detach().cpu().numpy()
    if arr.size == 1:
        return arr.item()
    return arr

if __name__ == "__main__":
    import sys

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Contrastive Predictive Coding')
    parser.add_argument('--train_mode', type=str, choices=['cpc', 'downstream', 'baseline', 'decoder', 'explain'],
                        help='Select mode. Possible: cpc, downstream, baseline, decoder')
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

    parser.add_argument('--forward_classes', type=int, default=41,
                        help="The number of possible output classes (only relevant for downstream)")

    parser.add_argument('--warmup_steps', type=int, default=0, help="The number of warmup steps")

    parser.add_argument('--batch_size', type=int, default=24, help="The batch size")

    parser.add_argument('--latent_size', type=int, default=768,
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

    args = parser.parse_args()
    main(args)
