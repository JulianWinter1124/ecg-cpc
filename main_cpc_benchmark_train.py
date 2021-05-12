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

import baseline_rnn_simplest_gru
import cpc_downstream_cnn
import cpc_downstream_twolinear
import cpc_encoder_as_strided
# import cpc_base
from architectures_cpc import cpc_autoregressive_v0, cpc_combined, cpc_downstream_only, cpc_encoder_v0, cpc_intersect, \
    cpc_predictor_v0
from util import store_models
from util.data import ecg_datasets2
from util.full_class_name import fullname
from util.metrics import training_metrics, baseline_losses as bl
from util.store_models import save_model_architecture, save_model_checkpoint, save_model_variables_text_only


def main(args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu_device)
    print(f'Device set to : {torch.cuda.current_device()}. Selected was {args.gpu_device}')
    torch.cuda.manual_seed(args.seed)
    print(f'Seed set to : {args.seed}.')

    print(f'Model outputpath: {args.out_path}')
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    # georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/georgia/WFDB',
    #                                                     window_size=args.crop_size, pad_to_size=args.crop_size, return_labels=True,
    #                                                     normalize_fn=ecg_datasets2.normalize_feature_scaling)
    # cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc_train/',
    #                                                        window_size=args.crop_size, pad_to_size=args.crop_size, return_labels=True,
    #                                                     normalize_fn=ecg_datasets2.normalize_feature_scaling)
    # cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc', window_size=args.crop_size,
    #                                                  pad_to_size=args.crop_size, return_labels=True,
    #                                                     normalize_fn=ecg_datasets2.normalize_feature_scaling)
    # ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/ptbxl/WFDB', window_size=args.crop_size,
    #                                                   pad_to_size=args.crop_size, return_labels=True,
    #                                                     normalize_fn=ecg_datasets2.normalize_feature_scaling)
    # nature = ecg_datasets2.ECGChallengeDatasetBaseline('/home/juwin106/data/nature', window_size=args.crop_size,
    #                                                   pad_to_size=args.crop_size, return_labels=True,
    #                                                     normalize_fn=ecg_datasets2.normalize_feature_scaling)

    georgia_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                        window_size=args.crop_size, pad_to_size=args.crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                           window_size=args.crop_size, pad_to_size=args.crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    cpsc2_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge', window_size=args.crop_size,
                                                     pad_to_size=args.crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    ptbxl_challenge = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge', window_size=args.crop_size,
                                                      pad_to_size=args.crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)
    nature = ecg_datasets2.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database', window_size=args.crop_size,
                                                      pad_to_size=args.crop_size, return_labels=True,
                                                        normalize_fn=ecg_datasets2.normalize_feature_scaling)


    if args.redo_splits:
        ecg_datasets2.filter_update_classes_by_count([georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature], min_count=20)
        print("Warning! Redoing splits!")
        ptbxl_challenge.random_train_split_with_class_count()
        cpsc_challenge.random_train_split_with_class_count()
        cpsc2_challenge.random_train_split_with_class_count()
        georgia_challenge.random_train_split_with_class_count()

    
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file()
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file()
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file()
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file()

    ecg_datasets2.filter_update_classes_by_count([nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train, cpsc2_val, t4], 1)
    print('Classes after last update', len(ptbxl_train.classes), ptbxl_train.classes)


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
            timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size), args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'
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
        # cpc_downstream_only.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=False, use_context=True, verbose=False
        # ),
        cpc_downstream_twolinear.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=True, use_context=True, verbose=False
        ),
        cpc_downstream_cnn.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=True, use_context=True, verbose=False
        )
    ]
    trained_model_dicts = [ #continue training for these in some way
        {'folder': 'models/11_05_21-16/architectures_cpc.cpc_combined.CPCCombined0',
         'model': None #Model will be loaded by method below
         },
        {'folder': 'models/11_05_21-16/architectures_cpc.cpc_combined.CPCCombined1',
         'model': None #Model will be loaded by method below
         },
        {'folder': 'models/11_05_21-16/architectures_cpc.cpc_combined.CPCCombined2',
         'model': None #Model will be loaded by method below
         },
        {'folder': 'models/11_05_21-16/architectures_cpc.cpc_combined.CPCCombined3',
         'model': None #Model will be loaded by method below
         },
    ]
    for model_i, trained_model_dict in enumerate(trained_model_dicts): #hack bad
        model_path = trained_model_dict['folder']
        model_files = store_models.extract_model_files_from_dir(model_path)
        for mfile in model_files:
            fm_fs, cp_fs = mfile
            fm_f = fm_fs[0]
            cp_f = sorted(cp_fs)[-1]
            model = store_models.load_model_architecture(fm_f)
            model, _, epoch = store_models.load_model_checkpoint(cp_f, model, optimizer=None, device_id=f'cuda:{args.gpu_device}')
            trained_model_dict['model'] = model #load model into dict
            break #only take first you find
    models = [
        # baseline_cnn_v7.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v8.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v9.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v10.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v11.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # #baseline_cnn_v12.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v13.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v14.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v15.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=False), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[0], freeze_cpc=True), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=True), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[2], downstream_models[0], freeze_cpc=True), 'will_pretrain':True, 'will_downtrain':False},
        # {'model':cpc_combined.CPCCombined(pretrain_models[3], downstream_models[0], freeze_cpc=True), 'will_pretrain':True, 'will_downtrain':False},
        # {'model': cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        #baseline_rnn.BaselineNet(in_channels=args.channels, out_channels=None, out_classes=args.forward_classes, verbose=False),
        # baseline_rnn_simplest_gru.BaselineNet(in_channels=args.channels, out_channels=None, out_classes=args.forward_classes, verbose=False),
    ]


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

    def pretrain(model_i, model):
        model_name = fullname(model)
        pretrain_fun = getattr(model, 'pretrain', None)
        if not callable(pretrain_fun): #this is not a CPC model!
            print(f'{model_name} is not a CPC model (needs to implement pretrain)... Skipping pretrain call')
            return
        output_path = os.path.join(args.out_path, model_name+str(model_i))
        print("Begin pretraining of {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))
        save_model_architecture(output_path, model, model_name)
        save_model_variables_text_only(output_path, model)
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
                #torch.cuda.empty_cache()
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

    def downstream(model_i, model):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name+str(model_i))
        print("Begin training of {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))
        save_model_architecture(output_path, model, model_name)
        save_model_variables_text_only(output_path, model)
        model.cuda()
        model.train()
        # init optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
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
                        loss = bl.binary_cross_entropy(pred=pred, y=labels)
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
        print("Finished model {}. Output saved to dir: {} Progress: {}/{}".format(model_name, output_path, model_i + 1, len(models)))

        del model  # delete and free
        torch.cuda.empty_cache()

    for model_i, model_dict in enumerate(models): #TODO: easily select what training is necessary!
        if type(model_dict) == dict:
            model = model_dict['model']
            if model_dict['will_pretrain']:
                pretrain(model_i, model)
            if model_dict['will_downtrain']:
                downstream(model_i, model)
        else: #assume its a model and the dev was to lazy to make it a dict
            model = model_dict
            pretrain(model_i, model)
            downstream(model_i, model)

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

    parser.add_argument('--pretrain_epochs', type=int, help='The number of Epochs to pretrain', default=100)

    parser.add_argument('--downstream_epochs', type=int, help='The number of Epochs to downtrain', default=100)

    parser.add_argument('--seed', type=int, help='The seed used', default=0)

    parser.add_argument('--forward_mode', help="The forward mode to be used.", default='context',
                        type=str)  # , choices=['context, latents, all']

    parser.add_argument('--out_path', help="The output directory for losses and models",
                        default='models/' + str(datetime.datetime.now().strftime("%d_%m_%y-%H")) + '-train', type=str)

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

    parser.add_argument('--crop_size', type=int, default=4500,
                        help="The size of the data that it is cropped to. If data is smaller than this number, data gets padded with zeros")

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
