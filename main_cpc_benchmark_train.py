import argparse
import datetime
import glob
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ChainDataset

from util import store_models
from util.metrics import training_metrics, baseline_losses as bl
#import cpc_base
from architectures_cpc import cpc_autoregressive_v0, cpc_combined, cpc_downstream_only, cpc_encoder_v0, cpc_intersect, cpc_predictor_v0

from util.data import ecg_datasets2
from util.full_class_name import fullname
from util.store_models import save_model_architecture, save_model_checkpoint


def main(args):
    np.random.seed(args.seed)

    #torch.cuda.set_device(args.gpu_device)
    print(args.out_path)
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    # train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
    #     '/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
    #     # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
    #     window_size=9500)
    # val_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
    #     '/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
    #     # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
    #     window_size=9500)
    # georgia = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/georgia/WFDB', window_size=4500, pad_to_size=4500, use_labels=True)
    # cpsc_train = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc_train', window_size=4500, pad_to_size=4500, use_labels=True)
    # cpsc = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc', window_size=4500, pad_to_size=4500, use_labels=True)
    # ptbxl = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/ptbxl/WFDB', window_size=4500, pad_to_size=4500, use_labels=True)

    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=4650, pad_to_size=4650, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=4650, pad_to_size=4650, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=4650,
                                                     pad_to_size=4650, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=4650,
                                                      pad_to_size=4650, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=4650,
                                                      pad_to_size=4650, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)

    georgia_challenge.merge_and_update_classes([georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature])

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

    

    pretrain_train_dataset = ChainDataset([nature, ptbxl_train, georgia_train, cpsc_train, cpsc2_train]) #CPC TRAIN
    pretrain_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val]) #CPC VAL
    downstream_train_dataset = ChainDataset([ptbxl_train, georgia_train, cpsc_train, cpsc2_train])
    downstream_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val])
    pretrain_models = [
        cpc_intersect.CPC(
            cpc_encoder_v0.Encoder(args.channels, args.latent_size),
            cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
            cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
            args.timesteps_in, args.timesteps_out, args.latent_size,
            timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='some'
        ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size), args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size), args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, verbose=False
        # ),
    ]
    downstream_models = [
        cpc_downstream_only.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=False, use_context=True, verbose=False
        ),
        # cpc_downstream_only.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # )
    ]
    combined_models = [ #TODO: give 'is_trained' param so you can easily switch if model needs to train
        cpc_combined.CPCCombined(pretrain_models[0], downstream_models[0]), #{'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[0], freeze_cpc=True), 'optimizer':None}
        #cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1])
    ]
    trained_combined_model_folders = [ #continue training for these
        #'models/18_03_21-18/architectures_cpc.cpc_combined.CPCCombined'
    ]
    for model_i, model_path in enumerate(trained_combined_model_folders): #hack bad
        model_arch_path = glob.glob(os.path.join(model_path, '*full_model.pt'))[0]
        model_checkpoint_path = glob.glob(os.path.join(model_path, '*checkpoint*.pt'))[-1]
        print('Found path to model architecture {} and checkpoint {}'.format(model_arch_path, model_checkpoint_path))
        model = store_models.load_model_architecture(model_arch_path)  # load correct class
        model, _, epoch = store_models.load_model_checkpoint(model_checkpoint_path, model, None)  # load model weights
        new_model = cpc_combined.CPCCombined(model.cpc_model, downstream_models[0])
        combined_models = [new_model] + combined_models #prepend to list

    pretrain_train_loaders = [
        DataLoader(pretrain_train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn)
    ]
    pretrain_val_loaders = [
        DataLoader(pretrain_val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn)
    ]
    downstream_train_loaders = [
        DataLoader(downstream_train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn)
    ]
    downstream_val_loaders = [
        DataLoader(downstream_val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [
        # Functions that take two tensors as argument and give score or list of score #TODO: maybe use dict with name
        # accuracy_metrics.fn_score_label,
        # accuracy_metrics.tn_score_label,
        # accuracy_metrics.tp_score_label,
        # accuracy_metrics.f1_score,
        # accuracy_metrics.micro_avg_recall_score,
        # accuracy_metrics.micro_avg_precision_score,
        # accuracy_metrics.accuracy,
        training_metrics.zero_fit_score,
        training_metrics.class_fit_score
        # accuracy_metrics.class_count_prediction,
        # accuracy_metrics.class_count_truth
    ]

    def pretrain():
        for model_i, model in enumerate(combined_models):
            model_name = fullname(model)
            output_path = os.path.join(args.out_path, model_name)
            print("Begin pretraining of {}. Output will  be saved to dir: {}".format(model_name, output_path))
            # Create dirs and model info
            Path(output_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
                cfg.write(str(args))
            save_model_architecture(output_path, model, model_name)
            model.cuda()
            model.train()
            # init optimizer
            optimizer = Adam(model.parameters(), lr=3e-4)
            metrics = defaultdict(lambda: defaultdict(list))
            for epoch in range(1, args.pretrain_epochs + 1):
                starttime = time.time()  # train
                for train_loader_i, train_loader in enumerate(pretrain_train_loaders):
                    for dataset_tuple in train_loader:
                        data, _ = dataset_tuple
                        data = data.float().cuda()
                        optimizer.zero_grad()
                        acc, loss, hidden = model.pretrain(data, y=None, hidden=None)
                        loss.backward()
                        optimizer.step()
                        # saving metrics
                        metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                        metrics[epoch]['trainacc'].append(parse_tensor_to_numpy_or_scalar(acc))
                        if args.dry_run:
                            break
                        del data, loss, hidden
                    print("\tFinished training dataset {}. Progress: {}/{}".format(train_loader_i, train_loader_i + 1,
                                                                                   len(pretrain_train_loaders)))
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    for val_loader_i, val_loader in enumerate(pretrain_val_loaders):  # validate
                        for dataset_tuple in val_loader:
                            data, _ = dataset_tuple
                            data = data.float().cuda()
                            acc, loss, hidden = model.pretrain(data, y=None, hidden=None)
                            # saving metrics
                            metrics[epoch]['valloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                            metrics[epoch]['valacc'].append(parse_tensor_to_numpy_or_scalar(acc))
                            if args.dry_run:
                                break
                        print("\tFinished vaildation dataset {}. Progress: {}/{}".format(val_loader_i, val_loader_i + 1,
                                                                                         len(pretrain_val_loaders)))
                        del data, loss, hidden

                elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
                metrics[epoch]['elapsed_time'].append(elapsed_time)
                print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f}. Avg train acc: {:.4f}. Avg val acc: {:.4f}. Elapsed time: {}".format(
                    epoch, args.pretrain_epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']),
                    np.mean(metrics[epoch]['trainacc']), np.mean(metrics[epoch]['valacc']),
                    elapsed_time))
                if args.dry_run:
                    break
            pickle_name = "pretrain-model-{}-epochs-{}.pickle".format(model_name, args.pretrain_epochs)
            # Saving metrics in pickle
            with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
                pickle.dump(dict(metrics), pick_file)
            # Save model + model weights + optimizer state
            save_model_checkpoint(output_path, epoch=args.pretrain_epochs, model=model, optimizer=optimizer, name=model_name)
            print("Finished model {}. Progress: {}/{}".format(model_name, model_i + 1, len(pretrain_models)))

            del model  # delete and free
            torch.cuda.empty_cache()

    def downstream():
        for model_i, model in enumerate(combined_models):
            model_name = fullname(model)
            output_path = os.path.join(args.out_path, model_name)
            print("Begin training of {}. Output will  be saved to dir: {}".format(model_name, output_path))
            # Create dirs and model info
            Path(output_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
                cfg.write(str(args))
            save_model_architecture(output_path, model, model_name)
            model.cuda()
            model.train()
            # init optimizer
            optimizer = Adam(model.parameters(), lr=3e-4)
            metrics = defaultdict(lambda: defaultdict(list))
            for epoch in range(1, args.downstream_epochs + 1):
                starttime = time.time()  # train
                for train_loader_i, train_loader in enumerate(downstream_train_loaders):
                    for dataset_tuple in train_loader:
                        data, labels = dataset_tuple
                        data = data.float().cuda()
                        labels = labels.float().cuda()
                        optimizer.zero_grad()
                        pred = model(data, y=None)  # makes model return prediction instead of loss
                        loss = bl.binary_cross_entropy(pred=pred, y=labels) #bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
                        loss.backward()
                        optimizer.step()
                        # saving metrics
                        metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                        with torch.no_grad():
                            for i, fn in enumerate(metric_functions):
                                metrics[epoch]['acc_' + str(i)].append(
                                    parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                        if args.dry_run:
                            break
                        del data, pred, labels, loss
                    print("\tFinished training dataset {}. Progress: {}/{}".format(train_loader_i, train_loader_i + 1,
                                                                                   len(downstream_train_loaders)))

                    torch.cuda.empty_cache()
                with torch.no_grad():
                    for val_loader_i, val_loader in enumerate(downstream_val_loaders):  # validate
                        for dataset_tuple in val_loader:
                            data, labels = dataset_tuple
                            data = data.float().cuda()
                            labels = labels.float().cuda()
                            pred = model(data, y=None)  # makes model return prediction instead of loss
                            loss = bl.MSE_loss(pred=pred, y=labels)
                            # saving metrics
                            metrics[epoch]['valloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                            for i, fn in enumerate(metric_functions):
                                metrics[epoch]['val_acc_' + str(i)].append(
                                    parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                            if args.dry_run:
                                break
                        print("\tFinished vaildation dataset {}. Progress: {}/{}".format(val_loader_i, val_loader_i + 1,
                                                                                         len(downstream_val_loaders)))
                        del data, pred, labels, loss

                elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
                metrics[epoch]['elapsed_time'].append(elapsed_time)
                print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f} Elapsed time: {}".format(
                    epoch, args.downstream_epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']),
                    elapsed_time))
                if args.dry_run:
                    break
            pickle_name = "model-{}-epochs-{}.pickle".format(model_name, args.downstream_epochs)
            # Saving metrics in pickle
            with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
                pickle.dump(dict(metrics), pick_file)
            # Save model + model weights + optimizer state
            save_model_checkpoint(output_path, epoch=args.downstream_epochs, model=model, optimizer=optimizer, name=model_name)
            print("Finished model {}. Progress: {}/{}".format(model_name, model_i + 1, len(combined_models)))

            del model  # delete and free
            torch.cuda.empty_cache()
    pretrain()
    downstream()

def parse_tensor_to_numpy_or_scalar(input_tensor):
    arr = input_tensor.detach().cpu().numpy()
    if arr.size == 1:
        return arr.item()
    return arr


if __name__ == "__main__":
    import sys

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Contrastive Predictive Coding')
    # datapath
    # Other params
    parser.add_argument('--saved_model', type=str,
                        help='Model path to load weights from. Has to be given for downstream mode.')

    parser.add_argument('--epochs', type=int, help='The number of Epochs to train', default=100)

    parser.add_argument('--pretrain_epochs', type=int, help='The number of Epochs to pretrain', default=100)

    parser.add_argument('--downstream_epochs', type=int, help='The number of Epochs to downtrain', default=100)

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

    parser.add_argument('--window_size', type=int, default=512,
                        help="The number of datapoints per channel per window")

    parser.add_argument('--hidden_size', type=int, default=512,
                        help="The size of the cell state/context used for predicting future latents or solving downstream tasks")

    parser.add_argument('--dry_run', dest='dry_run', action='store_true',
                        help="Only run minimal samples to test all models functionality")
    parser.set_defaults(dry_run=False)

    parser.add_argument('--redo_splits', dest='redo_splits', action='store_true',
                        help="Redo splits. Warning! File will be overwritten!")
    parser.set_defaults(redo_splits=False)

    parser.add_argument("--gpu_device", type=int, default=0)

    args = parser.parse_args()
    main(args)
