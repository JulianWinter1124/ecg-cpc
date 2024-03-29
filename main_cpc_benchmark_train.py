import argparse
import copy
import datetime
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ChainDataset

from architectures_baseline_challenge import baseline_cnn_v14, baseline_cnn_v15, baseline_cnn_v8, baseline_TCN_down, baseline_cnn_v2, \
    baseline_FCN, baseline_MLP, baseline_TCN_block, baseline_TCN_flatten, baseline_TCN_last, baseline_alex_v2, baseline_cnn_v0, baseline_cnn_v0_1, \
    baseline_cnn_v0_2, baseline_cnn_v0_3, baseline_cnn_v1, baseline_cnn_v3, baseline_cnn_v4, baseline_cnn_v5, baseline_cnn_v6, baseline_cnn_v7, \
    baseline_cnn_v9, baseline_resnet


from architectures_cpc import cpc_autoregressive_v0, cpc_combined, cpc_encoder_v0, cpc_intersect, \
    cpc_predictor_v0, cpc_encoder_as_strided, cpc_downstream_cnn, cpc_downstream_only, \
    cpc_downstream_latent_maximum, cpc_downstream_twolinear_v2, cpc_downstream_latent_average, \
    cpc_intersect_manylatents, cpc_encoder_small, cpc_autoregressive_hidden, cpc_encoder_likev8, \
    cpc_predictor_nocontext, cpc_predictor_nocontext


from util import store_models
from util.data import ecg_datasets3
from util.utility.full_class_name import fullname
from util.metrics import training_metrics, baseline_losses as bl
from util.store_models import save_model_architecture, save_model_checkpoint, save_model_variables_text_only, \
    save_model_state_dict_checkpoint


def main(args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu_device)
    print(f'Device set to : {torch.cuda.current_device()}. Selected was {args.gpu_device}')
    torch.cuda.manual_seed(args.seed)
    print(f'Seed set to : {args.seed}.')

    print(f'Model outputpath: {args.out_path}')
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    # norm_fn = ecg_datasets3.normalize_minmax_scaling_different
    if hasattr(ecg_datasets3, args.norm_fn):
        norm_fn = getattr(ecg_datasets3, args.norm_fn)
    else:
        norm_fn = ecg_datasets3.normalize_std_scaling

    # georgia_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/home/juwin106/data/georgia/WFDB',
    #                                                               window_size=args.crop_size,
    #                                                               pad_to_size=args.crop_size, return_labels=True,
    #                                                               normalize_fn=norm_fn)
    # cpsc_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc_train/',
    #                                                            window_size=args.crop_size, pad_to_size=args.crop_size,
    #                                                            return_labels=True,
    #                                                            normalize_fn=norm_fn)
    # cpsc2_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/home/juwin106/data/cpsc',
    #                                                             window_size=args.crop_size,
    #                                                             pad_to_size=args.crop_size, return_labels=True,
    #                                                             normalize_fn=norm_fn)
    # ptbxl_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/home/juwin106/data/ptbxl/WFDB',
    #                                                             window_size=args.crop_size,
    #                                                             pad_to_size=args.crop_size, return_labels=True,
    #                                                             normalize_fn=norm_fn)
    # nature = ecg_datasets3.ECGChallengeDatasetBaseline('/home/juwin106/data/nature',
    #                                                    window_size=args.crop_size,
    #                                                    pad_to_size=args.crop_size, return_labels=True,
    #                                                    normalize_fn=norm_fn)
    
    georgia_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/georgia_challenge/',
                                                                  window_size=args.crop_size,
                                                                  pad_to_size=args.crop_size, return_labels=True,
                                                                  normalize_fn=norm_fn)
    cpsc_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/cps2018_challenge/',
                                                               window_size=args.crop_size, pad_to_size=args.crop_size,
                                                               return_labels=True,
                                                               normalize_fn=norm_fn)
    cpsc2_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/china_challenge',
                                                                window_size=args.crop_size,
                                                                pad_to_size=args.crop_size, return_labels=True,
                                                                normalize_fn=norm_fn)
    ptbxl_challenge = ecg_datasets3.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/ptbxl_challenge',
                                                                window_size=args.crop_size,
                                                                pad_to_size=args.crop_size, return_labels=True,
                                                                normalize_fn=norm_fn)
    nature = ecg_datasets3.ECGChallengeDatasetBaseline('/media/julian/data/data/ECG/nature_database',
                                                       window_size=args.crop_size,
                                                       pad_to_size=args.crop_size, return_labels=True,
                                                       normalize_fn=norm_fn)

    if args.redo_splits:
        ecg_datasets3.filter_update_classes_by_count(
            [georgia_challenge, cpsc_challenge, ptbxl_challenge, cpsc2_challenge, nature], min_count=20)
        print("Warning! Redoing splits!")
        ptbxl_challenge.random_train_split_with_class_count()
        cpsc_challenge.random_train_split_with_class_count()
        cpsc2_challenge.random_train_split_with_class_count()
        georgia_challenge.random_train_split_with_class_count()
        
    print("Loading splits file:", args.splits_file)
    ptbxl_train, ptbxl_val, t1 = ptbxl_challenge.generate_datasets_from_split_file(args.splits_file)
    georgia_train, georgia_val, t2 = georgia_challenge.generate_datasets_from_split_file(args.splits_file)
    cpsc_train, cpsc_val, t3 = cpsc_challenge.generate_datasets_from_split_file(args.splits_file)
    cpsc2_train, cpsc2_val, t4 = cpsc2_challenge.generate_datasets_from_split_file(args.splits_file)

    ecg_datasets3.filter_update_classes_by_count(
        [nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train,
         cpsc2_val, t4], 1)
    print('Classes after last update', len(ptbxl_train.classes), ptbxl_train.classes)
    if args.use_class_weights:
        counts, counted_classes = ecg_datasets3.count_merged_classes(
            [nature, ptbxl_train, ptbxl_val, t1, georgia_train, georgia_val, t2, cpsc_train, cpsc_val, t3, cpsc2_train,
             cpsc2_val, t4])
        class_weights = torch.Tensor(max(counts) / counts).to(device=f'cuda:{args.gpu_device}')
        print('Using the following class weights:', class_weights)
    else:
        class_weights = None


    if args.preload_fraction > 0. and getattr(ptbxl_train, 'preload'):
        print("Preloading data...")
        ptbxl_train.preload(args.preload_fraction)
        ptbxl_val.preload(args.preload_fraction)
        georgia_train.preload(args.preload_fraction)
        georgia_val.preload(args.preload_fraction)
        cpsc_train.preload(args.preload_fraction)
        cpsc_val.preload(args.preload_fraction)
        cpsc2_train.preload(args.preload_fraction)
        cpsc2_val.preload(args.preload_fraction)

    pretrain_train_dataset = ChainDataset([nature, ptbxl_train, georgia_train, cpsc_train, cpsc2_train])  # CPC TRAIN
    pretrain_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val])  # CPC VAL
    downstream_train_dataset = ChainDataset([ptbxl_train, georgia_train, cpsc_train, cpsc2_train])
    downstream_val_dataset = ChainDataset([ptbxl_val, georgia_val, cpsc_val, cpsc2_val])
    pretrain_models = [
        # cpc_intersect.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in // 4),
        #     args.timesteps_in // 4, args.timesteps_out // 4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='multisame'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='multisame'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'
        # ),
        # cpc_intersect.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='future'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_small.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='all'
        # ),

        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1), #With hidden instead of context
        #     cpc_predictor_stacked.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1), #With hidden instead of context
        #     cpc_predictor_nocontext.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=True, verbose=False, sampling_mode='crossentropy-nocontext'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1), #With hidden instead of context
        #     cpc_predictor_nocontext.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=True, verbose=False, sampling_mode='crossentropy-nocontext'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_likev8.Encoder(args.channels, args.latent_size), args.window_size),
        #     cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1), #With hidden instead of context
        #     cpc_predictor_nocontext.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy-nocontext'
        # # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_small.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_nocontext.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy-nocontext'
        # ),

        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=True, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=True, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
        #     args.timesteps_in, args.timesteps_out, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        cpc_intersect_manylatents.CPC(
            cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
            cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1),
            cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_in),
            args.timesteps_in, args.timesteps_out, args.latent_size,
            timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_likev8.Encoder(args.channels, args.latent_size),
        #                                           window_size=545),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_hidden.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_v0.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy'
        # ),
        # cpc_intersect_manylatents.CPC(
        #     cpc_encoder_as_strided.StridedEncoder(cpc_encoder_v0.Encoder(args.channels, args.latent_size),
        #                                           args.window_size),
        #     cpc_autoregressive_v0.AutoRegressor(args.latent_size, args.hidden_size, 1),
        #     cpc_predictor_nocontext.Predictor(args.hidden_size, args.latent_size, args.timesteps_out//4),
        #     args.timesteps_in//4, args.timesteps_out//4, args.latent_size,
        #     timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='crossentropy-nocontext'
        # ),
    ]
    downstream_models = [
        # cpc_downstream_only.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=False, use_context=True, verbose=False
        # ),
        # cpc_downstream_only.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # ),
        # cpc_downstream_only.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=True, verbose=False
        # ),
        # cpc_downstream_cnn.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # ),
        cpc_downstream_latent_maximum.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=True, use_context=True, verbose=False
        ),
        cpc_downstream_latent_average.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=True, use_context=True, verbose=False
        ),
        # cpc_downstream_latent_maximum.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # ),
        # cpc_downstream_latent_average.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # ),
        cpc_downstream_twolinear_v2.DownstreamLinearNet(
            latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
            use_latents=True, use_context=True, verbose=False
        ),
        # cpc_downstream_twolinear_v2.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=True, use_context=False, verbose=False
        # ),
        # cpc_downstream_twolinear_v2.DownstreamLinearNet(
        #     latent_size=args.latent_size, context_size=args.hidden_size, out_classes=args.forward_classes,
        #     use_latents=False, use_context=True, verbose=False
        # ),
    ]
    trained_model_dicts = [  # continue training for these in some way
        # {
        #     'folder': "/home/julian/Downloads/Github/contrastive-predictive-coding/models/15_12_21-21-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|frozen|C|L|m:same|cpc_downstream_latent_maximum",
        #     'model': None  # standard same
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/15_12_21-21-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined1|train-test-splits|use_weights|strided|frozen|C|L|m:same|cpc_downstream_latent_maximum',
        #     'model': None  # standard same strided
        #     },
        # {
        #     'folder': "/home/julian/Downloads/Github/contrastive-predictive-coding/models/15_12_21-21-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined2|train-test-splits|use_weights|frozen|C|L|m:all|cpc_downstream_latent_maximum",
        #     'model': None  # standard all
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/15_12_21-21-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined3|train-test-splits|use_weights|strided|frozen|C|L|m:all|cpc_downstream_latent_maximum',
        #     'model': None  # standard all strided
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_02_22-16-12-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # many latents v0s
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_02_22-16-12-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined1|train-test-splits|use_weights|strided|unfrozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # many latents v0 strided
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/04_02_22-17-52-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|L|m:multisame|cpc_downstream_latent_maximum',
        #     'model': None  # normal multisame
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/04_02_22-17-52-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined1|train-test-splits|use_weights|unfrozen|C|L|m:same|cpc_downstream_latent_maximum',
        #     'model': None  # normal same
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/26_01_22-14-14-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # letent max cpc, down only 120
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_02_22-17-29-train|(6x)cpc/architectures_cpc.cpc_combined.CPCCombined3|train-test-splits|use_weights|frozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # latent max cpc, pretrained 100
        #     },
        # #### new archs test
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_02_22-16-12-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # v0
        #     },
        {
            'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_12_21-19-52-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|frozen|C|m:crossentropy|cpc_downstream_only',
            'model': None  # v8
            },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/22_12_21-12-train|cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|frozen|C|m:all|cpc_downstream_only',
        #     'model': None  # v0 hidden
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/07_02_22-12-07-train|cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|L|m:crossentropy|cpc_downstream_latent_maximum',
        #     'model': None  # v8 hidden
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/05_01_22-18-16-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|unfrozen|C|m:crossentropy-nocontext|cpc_downstream_only',
        #     'model': None  # v0 nocontext
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/04_01_22-16-51-train|cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|frozen|C|m:crossentropy-nocontext|cpc_downstream_only',
        #     'model': None  # v8 nocontext
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/27_01_22-14-49-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined0|train-test-splits|use_weights|strided|unfrozen|C|LNorm|m:crossentropy|cpc_downstream_only',
        #     'model': None  # strided normalized
        #     },
        # {
        #     'folder': '/home/julian/Downloads/Github/contrastive-predictive-coding/models/27_01_22-14-49-train|(2x)cpc/architectures_cpc.cpc_combined.CPCCombined1|train-test-splits|use_weights|unfrozen|C|LNorm|m:crossentropy|cpc_downstream_only',
        #     'model': None  #  normalized
        #     },
    ]
    for model_i, trained_model_dict in enumerate(trained_model_dicts):  # hack bad
        model_path = trained_model_dict['folder']
        model_files = store_models.extract_model_files_from_dir(model_path)
        if len(model_files) == 0:
            print(f"Could not find specified model {model_path}")
        for mfile in model_files:
            fm_fs, cp_fs, root = mfile
            fm_f = fm_fs[0]
            if not args.checkpoint_file_ending is None:
                temp = list(filter(lambda x: x.endswith(args.checkpoint_file_ending), cp_fs))
                if len(temp) == 1:
                    cp_f = temp[0]
                elif len(temp) > 1:
                    cp_f = sorted(cp_fs, key=lambda text: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)])[0]
                    print(f"WARNING! multiple checkpoint files fitting '{args.checkpoint_file_ending}': {temp}. Taking first")
                else:
                    print(f"WARNING! No files found matching {args.checkpoint_file_ending}. Selecting latest.")
                    cp_f = sorted(cp_fs, key=lambda text: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)])[-1]
            else:
                cp_f = sorted(cp_fs, key=lambda text: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)])[-1]
            model = store_models.load_model_architecture(fm_f)
            model, _, epoch = store_models.load_model_checkpoint(cp_f, model, optimizer=None,
                                                                 device_id=f'cuda:{args.gpu_device}')
            trained_model_dict['model'] = model  # load model into dict
            trained_model_dict['pretrained_epochs'] = epoch
            break  # only take first you find
    models = [

        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[1], freeze_cpc=False),
         'pretrained_epochs':trained_model_dicts[0]['pretrained_epochs'],
         'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[2], freeze_cpc=False),
         'pretrained_epochs':trained_model_dicts[0]['pretrained_epochs'],
         'will_pretrain': False, 'will_downtrain': True},

        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[1], freeze_cpc=True),
         'pretrained_epochs':trained_model_dicts[0]['pretrained_epochs'],
         'will_pretrain': False, 'will_downtrain': True},
        {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[2], freeze_cpc=True),
         'pretrained_epochs':trained_model_dicts[0]['pretrained_epochs'],
         'will_pretrain': False, 'will_downtrain': True},
        # ###
        # # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1], freeze_cpc=False),
        # #  'pretrained_epochs':trained_model_dicts[1]['pretrained_epochs'],
        # #  'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[2], freeze_cpc=False),
        #  'pretrained_epochs':trained_model_dicts[1]['pretrained_epochs'],
        #  'will_pretrain': False, 'will_downtrain': True},
        #
        # # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1], freeze_cpc=True),
        # #  'pretrained_epochs':trained_model_dicts[1]['pretrained_epochs'],
        # #  'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[2], freeze_cpc=True),
        #  'pretrained_epochs':trained_model_dicts[1]['pretrained_epochs'],
        #  'will_pretrain': False, 'will_downtrain': True},
        # ###
        # # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1], freeze_cpc=False),
        # #  'pretrained_epochs':trained_model_dicts[2]['pretrained_epochs'],
        # #  'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[2], freeze_cpc=False),
        #  'pretrained_epochs':trained_model_dicts[2]['pretrained_epochs'],
        #  'will_pretrain': False, 'will_downtrain': True},
        #
        # # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1], freeze_cpc=True),
        # #  'pretrained_epochs':trained_model_dicts[2]['pretrained_epochs'],
        # #  'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[2], freeze_cpc=True),
        #  'pretrained_epochs':trained_model_dicts[2]['pretrained_epochs'],
        #  'will_pretrain': False, 'will_downtrain': True}

        # {'model': cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=False), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(pretrain_models[1], downstream_models[1], freeze_cpc=False), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[0], freeze_cpc=False), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[0], freeze_cpc=False), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[0], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[0], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[0], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[0], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        #
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        #
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[5], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[5], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[5], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[5], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[4]['model'].cpc_model, downstream_models[0], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[2]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[2]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[2]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[2]), 'will_pretrain': False, 'will_downtrain': True},
        # baseline_FCN.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_MLP.BaselineNet(in_features=args.crop_size, out_classes=args.forward_classes, verbose=False),
        # baseline_alex_v2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_resnet.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v0.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v0_1.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v0_2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v0_3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v1.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v4.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v5.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # # baseline_cnn_v6.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v7.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v8.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v9.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_TCN_last.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # # baseline_TCN_flatten.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_TCN_down.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_TCN_block.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v14.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # baseline_cnn_v15.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=False), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[0], freeze_cpc=True), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=True), 'will_pretrain':False, 'will_downtrain':True},
        # {'model':cpc_combined.CPCCombined(pretrain_models[2], downstream_models[0], freeze_cpc=True), 'will_pretrain':True, 'will_downtrain':False},
        # {'model':cpc_combined.CPCCombined(pretrain_models[3], downstream_models[0], freeze_cpc=True), 'will_pretrain':True, 'will_downtrain':False},
        # {'model': cpc_combined.CPCCombined(pretrain_models[0], downstream_models[1], freeze_cpc=True), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True, 'desc':
        #  'blabla'},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[0]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[1]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[0]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[3]['model'].cpc_model, downstream_models[1]), 'will_pretrain': False, 'will_downtrain': True},
        # {'model': cpc_combined.CPCCombined(trained_model_dicts[2]['model'].cpc_model, downstream_models[2]), 'will_pretrain': False, 'will_downtrain': True},
        # baseline_rnn_simplest_lstm.BaselineNet(in_channels=args.channels, out_channels=None, out_classes=args.forward_classes, verbose=False),
        # baseline_rnn_simplest_gru.BaselineNet(in_channels=args.channels, out_channels=None, out_classes=args.forward_classes, verbose=False),
    ]

    pretrain_train_loaders = [
        DataLoader(pretrain_train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets3.collate_fn, pin_memory=True)
    ]
    pretrain_val_loaders = [
        DataLoader(pretrain_val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets3.collate_fn, pin_memory=True)
    ]
    downstream_train_loaders = [
        DataLoader(downstream_train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets3.collate_fn)
    ]
    downstream_val_loaders = [
        DataLoader(downstream_val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=1,
                   collate_fn=ecg_datasets3.collate_fn)
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
#         training_metrics.zero_fit_score,
#         training_metrics.class_fit_score
        # accuracy_metrics.class_count_prediction,
        # accuracy_metrics.class_count_truth
    ]

    def pretrain(model_i, model):
        model_name = model.name if hasattr(model, 'name') else fullname(model)
        pretrain_fun = getattr(model, 'pretrain', None)
        if not callable(pretrain_fun):  # this is not a CPC model!
            print(f'{model_name} is not a CPC model (needs to implement pretrain)... Skipping pretrain call')
            return
        output_path = os.path.join(args.out_path, model_name + str(model_i))
        print("Begin pretraining of {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            temp = args.downstream_epochs #copy
            args.downstream_epochs = 0 #change for save because model hasnt been downtrained yet and maybe is not
            cfg.write(str(args))
            args.downstream_epochs = temp
        save_model_architecture(output_path, model, model_name)
        save_model_variables_text_only(output_path, model)
        model.cuda()
        model.train()
        # init optimizer
        optimizer = Adam(model.parameters(), lr=3e-3)#3e-4
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=3e-10, verbose=True)
        metrics = defaultdict(lambda: defaultdict(list))
        update_count = 0
        train_mean_loss = torch.Tensor([0.]).cuda()
        train_mean_acc = torch.Tensor([0.]).cuda()
        val_mean_loss = torch.Tensor([0.]).cuda()
        val_mean_acc = torch.Tensor([0.]).cuda()
        for epoch in range(1, args.pretrain_epochs + 1):
            starttime = time.time()  # train
            moving_average = 0
            for train_loader_i, train_loader in enumerate(pretrain_train_loaders):
                for dataset_tuple in train_loader:
                    data, _ = dataset_tuple
                    data = data.float().cuda()
                    optimizer.zero_grad()
                    acc, loss, hidden = model.pretrain(data, y=None, hidden=None)
                    loss.backward()
                    optimizer.step()
                    update_count += 1
                    # saving metrics
                    if not args.no_metrics:
                        metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                        metrics[epoch]['trainacc'].append(parse_tensor_to_numpy_or_scalar(acc))
                    train_mean_loss += loss
                    train_mean_acc += acc
                    moving_average += 1
                    if args.dry_run:
                        break
                    del data, loss, hidden
                print(f"\tFinished training dataset {train_loader_i}. Progress: {train_loader_i + 1}/{len(pretrain_train_loaders)}")
                # torch.cuda.empty_cache()
            train_mean_loss = parse_tensor_to_numpy_or_scalar(train_mean_loss)/moving_average
            train_mean_acc = parse_tensor_to_numpy_or_scalar(train_mean_acc)/moving_average
            moving_average=0
            with torch.no_grad():
                for val_loader_i, val_loader in enumerate(pretrain_val_loaders):  # validate
                    for dataset_tuple in val_loader:
                        data, _ = dataset_tuple
                        data = data.float().cuda()
                        acc, loss, hidden = model.pretrain(data, y=None, hidden=None)
                        # saving metrics
                        if not args.no_metrics:
                            metrics[epoch]['valloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                            metrics[epoch]['valacc'].append(parse_tensor_to_numpy_or_scalar(acc))
                        val_mean_loss += loss
                        val_mean_acc += acc
                        moving_average += 1
                        if args.dry_run:
                            break
                    print("\tFinished vaildation dataset {}. Progress: {}/{}".format(val_loader_i, val_loader_i + 1,
                                                                                     len(pretrain_val_loaders)))
                    del data, loss, hidden
            val_mean_loss = parse_tensor_to_numpy_or_scalar(val_mean_loss)/moving_average
            val_mean_acc = parse_tensor_to_numpy_or_scalar(val_mean_acc)/moving_average
            scheduler.step(val_mean_loss)
            elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
            metrics[epoch]['elapsed_time'].append(elapsed_time)
            print(
                "Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f}. Avg train acc: {:.4f}. Avg val acc: {:.4f}. Elapsed time: {}. Total optimizer steps: {}.".format(
                    epoch, args.pretrain_epochs, train_mean_loss, val_mean_loss,
                    train_mean_acc, val_mean_acc, elapsed_time, update_count))
            if args.no_metrics:
                metrics[epoch]['trainloss'].append(train_mean_loss)
                metrics[epoch]['trainacc'].append(train_mean_acc)
                metrics[epoch]['valloss'].append(val_mean_loss)
                metrics[epoch]['valacc'].append(val_mean_acc)
            if args.dry_run:
                break
            if epoch in args.save_at_epoch_pre:
                save_model_checkpoint(output_path, epoch=epoch, model=model, optimizer=optimizer,
                              name=model_name)
        pickle_name = "pretrain-model-{}-epochs-{}.pickle".format(model_name, args.pretrain_epochs)
        # Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        # Save model + model weights + optimizer state
        save_model_checkpoint(output_path, epoch=args.pretrain_epochs, model=model, optimizer=optimizer,
                              name=model_name)
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i + 1, len(models)))

        del model  # delete and free
        torch.cuda.empty_cache()

    def downstream(model_i, model, pretrained_epochs=None):
        model_name = model.name if hasattr(model, 'name') else fullname(model)
        output_path = os.path.join(args.out_path, model_name + str(model_i))
        print("Begin training of {}. Output will  be saved to dir: {}".format(model_name, output_path))
        # Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)

        save_model_architecture(output_path, model, model_name)
        save_model_variables_text_only(output_path, model)
        model.cuda()
        model.train()
        # init optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=3e-10, verbose=True)
        best_val_model = None
        best_val_loss = None
        best_val_epoch = None
        metrics = defaultdict(lambda: defaultdict(list))
        update_count = 0

        train_mean_loss = torch.Tensor([0.]).cuda()
        val_mean_loss = torch.Tensor([0.]).cuda()
        moving_average = 0
        if args.downstream_updates_limit > 0:
            args.downstream_epochs = 99999999
        for epoch in range(1, args.downstream_epochs + 1):
            starttime = time.time()  # train
            for train_loader_i, train_loader in enumerate(downstream_train_loaders):
                for dataset_tuple in train_loader:
                    data, labels = dataset_tuple
                    data = data.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                    pred = model(data, y=None)  # makes model return prediction instead of loss
                    loss = bl.binary_cross_entropy(pred=pred, y=labels,
                                                   weight=class_weights)  # bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
                    loss.backward()
                    optimizer.step()
                    update_count += 1
                    train_mean_loss += loss
                    moving_average += 1
                    if not args.no_metrics:
                        with torch.no_grad():
                            for i, fn in enumerate(metric_functions):
                                metrics[epoch]['acc_' + str(i)].append(
                                    parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                        # saving metrics
                        metrics[epoch]['trainloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                    del data, pred, labels, loss
                    if args.dry_run:
                        break
                    if args.downstream_updates_limit > 0 and args.downstream_updates_limit <= update_count:
                        break
                print(f"\tFinished training dataset {train_loader_i}. Progress: {train_loader_i + 1}/{len(downstream_train_loaders)}")
                torch.cuda.empty_cache()
                if args.downstream_updates_limit > 0 and args.downstream_updates_limit <= update_count:
                    break
            train_mean_loss = parse_tensor_to_numpy_or_scalar(train_mean_loss)/moving_average
            moving_average=0
            with torch.no_grad():
                for val_loader_i, val_loader in enumerate(downstream_val_loaders):  # validate
                    for dataset_tuple in val_loader:
                        data, labels = dataset_tuple
                        data = data.float().cuda()
                        labels = labels.float().cuda()
                        pred = model(data, y=None)  # makes model return prediction instead of loss
                        loss = bl.binary_cross_entropy(pred=pred, y=labels, weight=class_weights)
                        val_mean_loss += loss
                        moving_average += 1
                        if not args.no_metrics:
                            for i, fn in enumerate(metric_functions):
                                metrics[epoch]['val_acc_' + str(i)].append(
                                    parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                            # saving metrics
                            metrics[epoch]['valloss'].append(parse_tensor_to_numpy_or_scalar(loss))
                        del data, pred, labels, loss
                        if args.dry_run:
                            break
            val_mean_loss = parse_tensor_to_numpy_or_scalar(val_mean_loss)/moving_average
            if args.no_metrics:
                metrics[epoch]['trainloss'].append(train_mean_loss)
                metrics[epoch]['valloss'].append(val_mean_loss)


                print("\tFinished vaildation dataset {}. Progress: {}/{}".format(val_loader_i, val_loader_i + 1,
                                                                                 len(downstream_val_loaders)))
            if epoch in args.save_at_epoch_down:
                save_model_checkpoint(output_path, epoch=epoch, model=model, optimizer=optimizer,
                              name=model_name)
            if best_val_loss is None or np.mean(metrics[epoch]['valloss']) < best_val_loss:
                best_val_loss = np.mean(metrics[epoch]['valloss'])
                best_val_model = copy.deepcopy(model.state_dict())
                best_val_epoch = epoch
            scheduler.step(np.mean(metrics[epoch]['valloss']))
            elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
            metrics[epoch]['elapsed_time'].append(elapsed_time)
            print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f} Elapsed time: {}. Total optimizer steps: {}.".format(
                epoch, args.downstream_epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']),
                elapsed_time, update_count))
            if args.dry_run:
                break
            if args.downstream_updates_limit > 0 and args.downstream_updates_limit <= update_count:
                break
        pickle_name = "model-{}-epochs-{}.pickle".format(model_name, epoch)
        # Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            if not pretrained_epochs is None:
                temp = args.pretrain_epochs #copy
                args.pretrain_epochs = pretrained_epochs #change for save
                cfg.write(str(args))
                args.pretrain_epochs = temp #Change back
            else:
                cfg.write(str(args))
        # Save model + model weights + optimizer state
        save_model_checkpoint(output_path, epoch=args.downstream_epochs, model=model, optimizer=optimizer,
                              name=model_name)
        save_model_state_dict_checkpoint(output_path, epoch=best_val_epoch, model_statedict=best_val_model, optimizer=None,
                              name=model_name, additional_name='_best_val_model')
        print("Finished model {}. Output saved to dir: {} Progress: {}/{}".format(model_name, output_path, model_i + 1,
                                                                                  len(models)))

        del model  # delete and free
        torch.cuda.empty_cache()

    print("Going to train", len(models), 'models')
    for model_i, model_dict in enumerate(models):  # TODO: easily select what training is necessary!
        if type(model_dict) == dict:
            model = copy.deepcopy(model_dict['model'])
            trained_epochs = None
            if 'desc' in model_dict:
                model.description = model_dict['desc']
            if 'name' in model_dict:
                model.name = model_dict['name']
            if 'pretrained_epochs' in model_dict:
                trained_epochs = model_dict['pretrained_epochs']
            elif 'will_pretrain' in model_dict and not model_dict['will_pretrain']: #its not an already trained model so...
                trained_epochs = 0
            if model_dict['will_pretrain']:
                pretrain(model_i, model)
            if model_dict['will_downtrain']:
                downstream(model_i, model, trained_epochs)
        else:  # assume its a model and the dev was to lazy to make it a dict
            model = model_dict
            pretrain(model_i, model)
            downstream(model_i, model)


def parse_tensor_to_numpy_or_scalar(input_tensor):
    try:
        return input_tensor.item()
    except:
        return input_tensor.detach().cpu().numpy()


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
                        default='models/' + str(datetime.datetime.now().strftime("%d_%m_%y-%H-%M")) + '-train', type=str)

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

    parser.add_argument('--use_class_weights', dest='use_class_weights', action='store_true',
                        help="Use class weights determined by datasets class count")
    parser.set_defaults(use_class_weights=False)
    
    parser.add_argument('--splits_file', type=str, default='train-test-splits.txt',
                        help="The Train val test split file to use")

    parser.add_argument('--norm_fn', type=str, default='normalize_std_scaling',
                        help="The Normalization function to use (from ecg_datasets3")

    parser.add_argument('--save_at_epoch_down', nargs='*', help='Selects additional downstream epochs to save the model weights at.', default=[], type=int)

    parser.add_argument('--save_at_epoch_pre', nargs='*', help='Selects additional pretraining epochs to save the model weights at.', default=[], type=int)

    parser.add_argument('--redo_splits', dest='redo_splits', action='store_true',
                        help="Redo splits. Warning! File will be overwritten!")
    parser.set_defaults(redo_splits=False)

    parser.add_argument('--no_metrics', dest='no_metrics', action='store_true',
                        help="No metrics (loss etc.) will be safed during training")
    parser.set_defaults(no_metrics=False)

    parser.add_argument('--checkpoint_file_ending', type=str, default=None)

    parser.add_argument("--gpu_device", type=int, default=0)

    parser.add_argument("--comment", type=str, default=None)

    parser.add_argument('--downstream_updates_limit', type=int, default=0)

    parser.add_argument('--downstream_updates_minimum', type=int, default=0)
    
    parser.add_argument("--preload_fraction", type=float, default=1.)

    args = parser.parse_args()
    main(args)
