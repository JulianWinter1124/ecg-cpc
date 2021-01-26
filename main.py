import argparse
import datetime
import os
import pickle

from pathlib import Path
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, ChainDataset

import baseline_cnn_explain
from baseline_architectures import baseline_convencoder, baseline_cnn_v0_3
import ecg_datasets2
#from cardio_model_small import CPC, Predictor, AutoRegressor, Encoder
#from cpc_architectures.cpc_encoder_vresnet import CPC, Predictor, AutoRegressor, Encoder
from cpc_architectures import cpc_encoder_v0, cpc_autoregressive_v0, cpc_predictor_v0, cpc_base
from optimizer import ScheduledOptim
from training import cpc_train, cpc_validation, down_train, down_validation, baseline_train, baseline_validation, \
    decoder_validation, decoder_train

import torch

from cpc_architectures.downstream_model_multitarget import DownstreamLinearNet
from util.full_class_name import fullname
from util.ptbxl_data import PTBXLData
from util.temporal_to_image_converter import timeseries_to_image


def main(args):

    np.random.seed(args.seed)

    # TODO: Put these params in the arguments as well
    # paramas

    timesteps_in = args.timesteps_in
    timesteps_out = args.timesteps_in
    number_of_latents = args.latent_size
    channels = args.channels
    train_batch_size = args.batch_size
    validation_batch_size = args.batch_size
    window_length = args.window_length  # Total size = window_length*n_windows
    hidden_size = args.hidden_size
    n_windows = timesteps_in + timesteps_out

    epochs = args.epochs

    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, 'params.txt'), 'w') as cfg:
        cfg.write('\n'.join([str(pa) for pa in [timesteps_in, timesteps_out, number_of_latents, channels, train_batch_size, validation_batch_size, window_length, n_windows, args]]))


    if args.train_mode == 'cpc':

        enc = cpc_encoder_v0.Encoder(channels=channels, latent_size=args.latent_size)  # TODO: automatically fit this to data
        enc.cuda()
        print(enc)

        auto = cpc_autoregressive_v0.AutoRegressor(n_latents=args.latent_size, hidden_size=hidden_size)
        auto.cuda()
        print(auto)

        predictor = cpc_predictor_v0.Predictor(hidden_size, args.latent_size, timesteps_out)
        predictor.cuda()
        print(predictor)

        model = cpc_base.CPC(enc, auto, predictor, args.latent_size, timesteps_in=timesteps_in, timesteps_out=timesteps_out)
        #model.
        #train_dataset_mit = ecg_datasets2.ECGDataset('/media/julian/Volume/data/ECG/mit-bih-arrhythmia-database-1.0.0/generated/resampled/train', window_size=window_length, n_windows=n_windows)
        #train_dataset_peters = ecg_datasets2.ECGDataset('/media/julian/Volume/data/ECG/st-petersburg-arrythmia-annotations/resampled/train', window_size=window_length, n_windows=n_windows, preload_windows=40)
        train_dataset_ptb = ecg_datasets2.ECGDatasetBatching('/media/julian/data/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized/train', window_size=window_length, n_windows=n_windows, preload_windows=40, batch_size=train_batch_size)
        train_dataset_ptbxl = ecg_datasets2.ECGDatasetBatching('/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train', window_size=window_length, n_windows=n_windows, preload_windows=40)
        train_dataset_sinus = ecg_datasets2.ECGDatasetBatching('/media/julian/data/data/sinus/train', window_size=window_length, n_windows=n_windows, preload_windows=40)

        #trainset_total = ecg_datasets2.ECGDatasetMultiple([train_dataset_peters, train_dataset_ptb, train_dataset_ptbxl]) #train_dataset_mit,
        #trainset_total = ecg_datasets2.ECGDatasetMultiple([train_dataset_ptb, train_dataset_ptbxl])
        trainset_total = ChainDataset([train_dataset_ptbxl])#
        #trainset_total = ChainDataset([train_dataset_sinus])
        dataloader = DataLoader(trainset_total, batch_size=train_batch_size, drop_last=True, num_workers=1)

        #val_dataset_mit = ecg_datasets2.ECGDataset('/media/julian/Volume/data/ECG/mit-bih-arrhythmia-database-1.0.0/generated/resampled/val', window_size=window_length, n_windows=n_windows)
        #val_dataset_peters = ecg_datasets2.ECGDataset('/media/julian/Volume/data/ECG/st-petersburg-arrythmia-annotations/resampled/val', window_size=window_length, n_windows=n_windows, preload_windows=40)
        #val_dataset_ptb = ecg_datasets2.ECGDatasetBatching('/media/julian/data/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized/val', window_size=window_length, n_windows=n_windows, preload_windows=40, batch_size=validation_batch_size)
        val_dataset_ptbxl = ecg_datasets2.ECGDataset('/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val', window_size=window_length, n_windows=n_windows, preload_windows=40)
        val_dataset_sinus = ecg_datasets2.ECGDatasetBatching('/media/julian/data/data/sinus/val', window_size=window_length, n_windows=n_windows, preload_windows=40)
        #valset_total = ecg_datasets2.ECGDatasetMultiple([val_dataset_ptb, val_dataset_ptbxl]) #val_dataset_mit,
        valset_total = ChainDataset([val_dataset_ptbxl])
        #valset_total = ChainDataset([val_dataset_sinus])#, val_dataset_ptbxl
        valloader = DataLoader(valset_total, batch_size=validation_batch_size, drop_last=True, num_workers=1)

        #libridataset = torchaudio.datasets.LIBRISPEECH("/home/julian/Datasets/", url='/home/julian/Datasets/LibriSpeech/train-clean-100/', folder_in_archive='LibriSpeech/train-clean-100/', download=False)
        #dataloader = DataLoader(libridataset, batch_size=8, collate_fn=lambda x: torch.stack([a[0] for a in x]))

        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)

        if args.saved_model:
            model, optimizer, epoch = load_model_state(out_path, model, optimizer)
        else:
            torch.save(model, os.path.join(out_path, "cpc_model_full.pt"))
        model.cuda()

        #model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ## Start training
        best_acc = 0
        best_loss = np.inf
        best_epoch = -1
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(1, epochs + 1):

            train_acc, train_loss = cpc_train(model, dataloader, timesteps_in, timesteps_out, optimizer, epoch, train_batch_size)
            val_acc, val_loss = cpc_validation(model, valloader, timesteps_in, timesteps_out, validation_batch_size)
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())
            val_accuracies.append(val_acc.item())
            # Save
            if val_loss < best_loss:  # TODO: maybe use accuracy (not sure if accuracy is a good measurement)
                best_loss = val_loss
                best_acc = max(val_acc, best_acc)
                best_epoch = epoch
                print("saving model")
                torch.save(model.state_dict(), os.path.join(out_path, "cpc_model_validation.pt"))
            if epoch - best_epoch >= 5:
                #update learning rate
                pass
                if epoch % 10 == 0:
                    save_model_state(out_path, epoch, args.train_mode, model, optimizer,
                                     [train_accuracies, val_accuracies],
                                     [train_losses, val_losses])
        save_model_state(out_path, epochs, args.train_mode, model, optimizer,
                             [train_accuracies, val_accuracies], [train_losses, val_losses])

    #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    if args.train_mode == 'downstream':

        #train_dataset_ptb = ecg_datasets2.ECGDatasetBatching('/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized/train', window_size=window_length, n_windows=n_windows, channels=slice(0,12), preload_windows=40, batch_size=train_batch_size, use_labels=True)
        train_dataset_ptbxl = ecg_datasets2.ECGDatasetBatching('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train', window_size=window_length, n_windows=n_windows, channels=slice(0,12), preload_windows=40, batch_size=train_batch_size, use_labels=True)
        trainset_total = ecg_datasets2.ECGMultipleDatasets([train_dataset_ptbxl])
        trainloader = DataLoader(trainset_total, batch_size=train_batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)

        #val_dataset_ptb = ecg_datasets2.ECGDatasetBatching('/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized/val',window_size=window_length, n_windows=n_windows, channels=slice(0, 12), preload_windows=40, batch_size=validation_batch_size, use_labels=True)
        val_dataset_ptbxl = ecg_datasets2.ECGDatasetBatching('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val', window_size=window_length, n_windows=n_windows, channels=slice(0,12), preload_windows=40, batch_size=train_batch_size, use_labels=True)
        # valset_total = ecg_datasets2.ECGDatasetMultiple([val_dataset_ptb, val_dataset_ptbxl]) #val_dataset_mit,
        valset_total = ecg_datasets2.ECGMultipleDatasets([val_dataset_ptbxl])
        valloader = DataLoader(valset_total, batch_size=validation_batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)

        model.load_state_dict(torch.load(args.saved_model)) #Load the trained cpc model
        model.eval()
        #model.freeze_layers()
        model.cuda()
        print(model)
        f_m = args.forward_mode
        downstream_model = DownstreamLinearNet(cpc_model_trained=model, timesteps_in=timesteps_in, context_size=hidden_size, latent_size=number_of_latents, out_classes=args.forward_classes, use_context=f_m == "context" or f_m == "all", use_latents=f_m == "latents" or f_m == "all")
        torch.save(downstream_model, os.path.join(out_path, "downstream_model_full.pt"))
        downstream_model.cuda()
        print()
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, downstream_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)

        best_acc = 0
        best_loss = np.inf
        best_epoch = -1
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(1, epochs + 1):

            train_acc, train_loss = down_train(downstream_model, trainloader, timesteps_in, timesteps_out, optimizer, epoch, None)
            val_acc, val_loss = down_validation(downstream_model, valloader, timesteps_in, timesteps_out, None)
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())
            val_accuracies.append(val_acc.item())
            # Save
            if val_loss < best_loss:  # TODO: maybe use accuracy (not sure if accuracy is a good measurement)
                best_loss = val_loss
                best_acc = max(val_acc, best_acc)
                best_epoch = epoch
                print("saving model")
                torch.save(downstream_model.state_dict(), os.path.join(out_path, "cpc_model_downstream_validation.pt"))
            if epoch - best_epoch >= 5:
                # update learning rate
                optimizer.increase_delta()
                best_epoch = epoch
                if epoch % 10 == 0:
                    save_model_state(out_path, epoch, args.train_mode, model, optimizer,
                                     [train_accuracies, val_accuracies],
                                     [train_losses, val_losses])
        save_model_state(out_path, epochs, args.train_mode, model, optimizer,
                             [train_accuracies, val_accuracies], [train_losses, val_losses])

    if args.train_mode == 'test':

        test_dataset_ptb = ecg_datasets2.ECGDatasetBatching(
            '/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized/test',
            window_size=window_length, n_windows=n_windows, channels=slice(0, 12), preload_windows=40,
            batch_size=validation_batch_size, use_labels=True)

        testset_total = ecg_datasets2.ECGMultipleDatasets([test_dataset_ptb])
        testloader = DataLoader(testset_total, batch_size=validation_batch_size, drop_last=True, num_workers=1,
                               collate_fn=ecg_datasets2.collate_fn)

        model.load_state_dict(torch.load(args.saved_model))  # Load the trained cpc model
        model.eval()
        model.freeze_layers()
        model.cuda()
        f_m = args.forward_mode
        downstream_model = DownstreamLinearNet(cpc_model_trained=model, timesteps_in=timesteps_in,
                                               context_size=hidden_size, latent_size=number_of_latents,
                                               out_classes=args.forward_classes,
                                               use_context=f_m == "context" or f_m == "all",
                                               use_latents=f_m == "latents" or f_m == "all")
        downstream_model.cuda()
        test_acc, test_loss = down_validation(downstream_model, testloader, timesteps_in, timesteps_out, None)


    if args.train_mode == 'baseline':
        train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti('/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',#'/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
            window_size=9500)

        trainloader = DataLoader(train_dataset_ptbxl, batch_size=train_batch_size, drop_last=True, num_workers=1,
                                 collate_fn=ecg_datasets2.collate_fn)

        val_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti('/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val', #'/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
            window_size=9500)
        valloader = DataLoader(val_dataset_ptbxl, batch_size=validation_batch_size, drop_last=True, num_workers=1,
                               collate_fn=ecg_datasets2.collate_fn)

        model = baseline_cnn_v0_3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False)

        print(model)
        model.cuda()
        with open(os.path.join(args.out_path, 'model_arch.txt'), 'w') as f:
            print(fullname(model), file=f)
            print(model, file=f)


        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)

        start_epoch = 0
        if args.saved_model:
            model, optimizer, start_epoch = load_model_state(args.saved_model, model, optimizer)
        else:
            torch.save(model, os.path.join(out_path, 'full_model.pt'))

        best_acc = 0
        best_loss = np.inf
        best_epoch = -1
        train_losses = []
        val_losses = []
        train_accuracies = []

        val_accuracies = []

        for epoch in range(start_epoch+1, epochs + start_epoch + 1):

            train_acc, train_loss = baseline_train(model, trainloader, optimizer, epoch, args)
            val_acc, val_loss = baseline_validation(model, valloader, optimizer, epoch, args)
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())
            val_accuracies.append(val_acc.item())
            # Save
            if val_acc < best_acc:  # TODO: maybe use accuracy (not sure if accuracy is a good measurement)
                best_acc = max(val_acc, best_acc)
                best_epoch = epoch
            if epoch - best_epoch >= 5:
                pass
                #optimizer.update_learning_rate()
                #best_epoch = epoch
            if epoch % 10 == 0:
                save_model_state(out_path, epoch, args.train_mode, model, optimizer, [train_accuracies, val_accuracies],
                                 [train_losses, val_losses])
        save_model_state(out_path, epochs, args.train_mode, model, optimizer,
                                 [train_accuracies, val_accuracies], [train_losses, val_losses])

    if args.train_mode == 'explain':

        test_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/test', #'/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
            window_size=9500)
        totalset = ChainDataset([test_dataset_ptbxl])#[train_dataset_ptbxl, val_dataset_ptbxl])
        dataloader = DataLoader(totalset, batch_size=validation_batch_size, drop_last=True, num_workers=1,
                               collate_fn=ecg_datasets2.collate_fn)

        bmodel, optimizer, epoch = load_model_state(os.path.join(os.path.split(args.saved_model)[0], 'full_model.pt'))
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, bmodel.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)

        bmodel, optimizer, epoch = load_model_state(args.saved_model, bmodel, optimizer)

        model = baseline_cnn_explain.ExplainLabel(bmodel)
        model.eval()
        model.cuda()

        ecg = PTBXLData(base_directory='/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/')
        ecg.init_multilabel_encoder()
        print(ecg.code_list)

        print(model)
        #vw = VideoWriter('images/ptbxl.mpeg', fps=2.0)
        model.train()
        for batch_idx, data_and_labels in enumerate(dataloader):
            data, labels = data_and_labels
            data = data.float().cuda()
            optimizer.zero_grad()
            output, grad = model(data, y=None)
            output = output.detach().cpu()
            top3 = np.argsort(output)[:, -3:]
            #TODO: Iterate over output != 0 and calc grad respectively
            pred_prob = []
            for i, t3 in enumerate(top3):
                pred_prob.append(list(zip([ecg.code_list[t] for t in t3], output[i, t3])))
            not0 = [[i for i, e in enumerate(a) if e != 0] for a in labels]

            ground_truth = []
            for i, t3 in enumerate(not0):
                ground_truth.append(list(zip([ecg.code_list[t] for t in t3], labels[i, t3])))
            img = timeseries_to_image(grad.cpu(), grad.cpu(), downsample_factor=5, convert_to_rgb=False, pred_classes=pred_prob, ground_truth=ground_truth, filename='images/ptbxl/gradient/ptbxl_timeseries_' + str(batch_idx), show=False)
            #write_video('test.mkv', img, 1.0) broken
            #vw.tensor_to_video_continuous(img)
        #vw.close()



    if args.train_mode == 'decoder':
        start_epoch = 1
        # train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',#'/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        #     window_size=9500)
        train_dataset_ptbxl = ecg_datasets2.ECGDatasetBatching(
            '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
            window_size=args.window_length, n_windows=1, preload_windows=40)

        trainloader = DataLoader(train_dataset_ptbxl, batch_size=train_batch_size, drop_last=False,
                                 collate_fn=ecg_datasets2.collate_fn)

        #val_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val', #'/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
        #    window_size=9500)
        val_dataset_ptbxl = ecg_datasets2.ECGDatasetBatching('/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val', window_size=args.window_length, n_windows=1, preload_windows=40)
        
        valloader = DataLoader(val_dataset_ptbxl, batch_size=validation_batch_size, drop_last=False,
                               collate_fn=ecg_datasets2.collate_fn)

        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)

        model = baseline_convencoder.BaselineNet(args.channels, args.latent_size, args.forward_classes)
        if args.saved_model:
            model, optimizer, start_epoch = load_model_state(args.saved_model, model, optimizer)

        model.cuda()

        print(model)

        best_acc = 0
        best_loss = np.inf
        best_epoch = -1
        train_losses = []
        val_losses = []
        train_accuracies = []

        val_accuracies = []

        for epoch in range(start_epoch, epochs + 1):

            train_acc, train_loss = decoder_train(model, trainloader, optimizer, epoch)
            val_acc, val_loss = decoder_validation(model, valloader, optimizer, epoch)
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())
            val_accuracies.append(val_acc.item())
            # Save
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = max(val_acc, best_acc)
                best_epoch = epoch
            if epoch - best_epoch >= 5:
                # update learning rate
                optimizer.increase_delta()
                best_epoch = epoch
            if epoch % 10 == 0:
                save_model_state(out_path, epoch, args.train_mode, model, optimizer, [train_accuracies, val_accuracies], [train_losses, val_losses])
        save_model_state(out_path, epochs, args.train_mode, model, optimizer, [train_accuracies, val_accuracies], [train_losses, val_losses])

def save_model_state(output_path, epoch, train_mode='', model=None, optimizer=None, accuracies=None, losses=None, full=False):
    if full:
        print("Saving full model...")
        name = 'model_full.pt'
        torch.save(model, os.path.join(output_path, name))
    else:
        print("saving model at epoch:", epoch)
        if not (model is None and optimizer is None):
            name = train_mode + '_modelstate_epoch' + str(epoch) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(output_path, name))
        if not (accuracies is None and losses is None):
            with open(os.path.join(output_path, 'losses.pkl'), 'wb') as pickle_file:
                pickle.dump(losses, pickle_file)
            with open(os.path.join(output_path, 'accuracies.pkl'), 'wb') as pickle_file:
                pickle.dump(accuracies, pickle_file)

def load_model_state(model_path, model=None, optimizer=None):
    if model is None:
        model = torch.load(model_path)
        epoch = 1
    else:
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
            epoch = 1

    model.eval()
    return model, optimizer, epoch




if __name__ == "__main__":
    import sys

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Contrastive Predictive Coding')
    parser.add_argument('--train_mode', type=str, choices=['cpc', 'downstream', 'baseline', 'decoder', 'explain'], help='Select mode. Possible: cpc, downstream, baseline, decoder')
    #datapath
    #Other params
    parser.add_argument('--saved_model', type=str,
                        help='Model path to load weights from. Has to be given for downstream mode.')

    parser.add_argument('--epochs', type=int, help='The number of Epochs to train', default=100)

    parser.add_argument('--seed', type=int, help='The seed used', default=None)

    parser.add_argument('--forward_mode', help="The forward mode to be used.", default='context', type=str) #, choices=['context, latents, all']

    parser.add_argument('--out_path', help="The output directory for losses and models", default='models/' + str(datetime.datetime.now().strftime("%d_%m_%y-%H")), type=str)

    parser.add_argument('--forward_classes', type=int, default=41, help="The number of possible output classes (only relevant for downstream)")

    parser.add_argument('--warmup_steps', type=int, default=0, help="The number of warmup steps")

    parser.add_argument('--batch_size', type=int, default=24, help="The batch size")

    parser.add_argument('--latent_size', type=int, default=768,
                        help="The size of the latent encoding for one window")

    parser.add_argument('--timesteps_in', type=int, default=6, help="The number of windows being used to form a context for prediction")

    parser.add_argument('--timesteps_out', type=int, default=6,
                        help="The number of windows being predicted from the context (cpc task exclusive)")

    parser.add_argument('--channels', type=int, default=12,
                        help="The number of channels the data will have") #TODO: auto detect

    parser.add_argument('--window_length', type=int, default=512,
                        help="The number of datapoints per channel per window")

    parser.add_argument('--hidden_size', type=int, default=512,
                        help="The size of the cell state/context used for predicting future latents or solving downstream tasks")

    parser.add_argument('--grad_clip', type=int, default=0.0,
                        help="The number where to clip gradients at (useful if they become nan")

    args = parser.parse_args()
    main(args)