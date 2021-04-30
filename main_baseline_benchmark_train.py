import argparse
import datetime
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ChainDataset
from architectures_baseline_challenge import baseline_cnn_v9
from util.metrics import training_metrics, baseline_losses as bl
from util.data import ecg_datasets2
from util.full_class_name import fullname
from util.store_models import save_model_architecture, save_model_checkpoint



def main(args):
    np.random.seed(args.seed)
    
    torch.cuda.set_device(args.gpu_device)
    print(args.out_path)
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                                  window_size=4650, pad_to_size=4650,
                                                                  return_labels=True,
                                                                  normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                               window_size=4650, pad_to_size=4650, return_labels=True,
                                                               normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge',
                                                                window_size=4650,
                                                                pad_to_size=4650, return_labels=True,
                                                                normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge',
                                                                window_size=4650,
                                                                pad_to_size=4650, return_labels=True,
                                                                normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=4650,
                                                       pad_to_size=4650, return_labels=True,
                                                       normalize_fn=ecg_datasets2.normalize_feature_scaling)

    georgia_challenge.merge_and_update_classes(
        [georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature])


    if args.redo_splits:
        print("Warning! Redoing splits!")
        ptbxl_challenge.random_train_split()
        cpsc_challenge.random_train_split()
        cpsc2_challenge.random_train_split()
        georgia_challenge.random_train_split()

    ptbxl_train, ptbxl_val, _ = ptbxl_challenge.generate_datasets_from_split_file()
    georgia_train, georgia_val, _ = georgia_challenge.generate_datasets_from_split_file()
    cpsc_train, cpsc_val, _ = cpsc_challenge.generate_datasets_from_split_file()
    cpsc2_train, cpsc2_val, _ = cpsc2_challenge.generate_datasets_from_split_file()

    pretrain_train_dataset = ChainDataset([nature, ptbxl_train, georgia_train, cpsc_train, cpsc2_train])  # CPC TRAIN
    pretrain_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val])  # CPC VAL
    downstream_train_dataset = ChainDataset([ptbxl_train, georgia_train, cpsc_train, cpsc2_train])
    downstream_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val])
    model_classes = [
        #baseline_cnn_v0.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v0_1.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v0_2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v0_3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v1.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v4.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v5.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v6.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v7.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v8.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v9.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v10.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v11.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # #baseline_cnn_v12.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # #baseline_cnn_v13.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v14.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        #baseline_cnn_v15.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False)
    ]

    train_loaders = [
        #DataLoader(train_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
        DataLoader(train_dataset_challenge, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
    ]
    val_loaders = [
        #DataLoader(val_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
        DataLoader(val_dataset_challenge, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score #TODO: maybe use dict with name
        #accuracy_metrics.fn_score_label,
        #accuracy_metrics.tn_score_label,
        #accuracy_metrics.tp_score_label,
        #accuracy_metrics.f1_score,
        # accuracy_metrics.micro_avg_recall_score,
        # accuracy_metrics.micro_avg_precision_score,
        # accuracy_metrics.accuracy,
        training_metrics.zero_fit_score,
        training_metrics.class_fit_score
        # accuracy_metrics.class_count_prediction,
        # accuracy_metrics.class_count_truth
    ]
    for model_i, model in enumerate(model_classes):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name)
        print("Begin training of {}. Output will  be saved to dir: {}".format(model_name, output_path))
        #Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))
        save_model_architecture(output_path, model, model_name)
        model.cuda()
        model.train()
        #init optimizer
        optimizer = Adam(model.parameters(), lr=3e-4)
        metrics = defaultdict(lambda: defaultdict(list))
        for epoch in range(1, args.epochs+1):
            starttime = time.time() #train
            for train_loader_i, train_loader in enumerate(train_loaders):
                for dataset_tuple in train_loader:
                    data, labels = dataset_tuple
                    data = data.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                    pred = model(data, y=None) #makes model return prediction instead of loss
                    loss = bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
                    loss.backward()
                    optimizer.step()
                    #saving metrics
                    metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                    with torch.no_grad():
                        for i, fn in enumerate(metric_functions):
                            metrics[epoch]['acc_'+str(i)].append(parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                    if args.dry_run:
                        break
                    del data, pred, labels, loss
                print("\tFinished training dataset {}. Progress: {}/{}".format(train_loader_i, train_loader_i + 1, len(train_loaders)))

                torch.cuda.empty_cache()
            with torch.no_grad():
                for val_loader_i, val_loader in enumerate(val_loaders): #validate
                    for dataset_tuple in val_loader:
                        data, labels = dataset_tuple
                        data = data.float().cuda()
                        labels = labels.float().cuda()
                        pred = model(data, y=None) #makes model return prediction instead of loss
                        loss = bl.MSE_loss(pred=pred, y=labels)
                        #saving metrics
                        metrics[epoch]['valloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                        for i, fn in enumerate(metric_functions):
                            metrics[epoch]['val_acc_'+str(i)].append(parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                        if args.dry_run:
                            break
                    print("\tFinished vaildation dataset {}. Progress: {}/{}".format(val_loader_i, val_loader_i + 1, len(val_loaders)))
                    del data, pred, labels, loss

            elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
            metrics[epoch]['elapsed_time'].append(elapsed_time)
            print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f} Elapsed time: {}".format(
                epoch, args.epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']), elapsed_time))
            if args.dry_run:
                break
        pickle_name = "model-{}-epochs-{}.pickle".format(model_name, args.epochs)
        #Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        #Save model + model weights + optimizer state
        save_model_checkpoint(output_path, epoch=args.epochs, model=model, optimizer=optimizer, name=model_name)
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i+1, len(model_classes)))

        del model #delete and free
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

    parser.add_argument('--forward_classes', type=int, default=52,
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

    parser.add_argument('--dry_run', dest='dry_run', action='store_true',
                        help="Only run minimal samples to test all models functionality")
    parser.set_defaults(dry_run=False)
    
    parser.add_argument("--gpu_device", type=int, default=0)

    args = parser.parse_args()
    main(args)
