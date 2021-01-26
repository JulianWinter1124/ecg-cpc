import argparse
import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from baseline_architectures import baseline_cnn_v2
import ecg_datasets2
from optimizer import ScheduledOptim
from training import baseline_train, baseline_validation


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
    
    

    models = [
        baseline_cnn_v2.BaselineNet(args.channels, args.forward_classes, verbose=False)
    ]
    for i, model in enumerate(models):
        torch.save(model, os.path.join(args.out_path, str(i)+'_model_full.pt'))
        with open(os.path.join(args.out_path, str(i)+'_model_arch.txt'), 'w') as f:
            print(model, file=f)

   
    
    for model_i, model in enumerate(models):
        model.cuda()
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            args.warmup_steps)
    
        best_acc = 0
        best_loss = np.inf
        best_epoch = -1
        train_losses = []
        val_losses = []
        train_accuracies = []
    
        val_accuracies = []
    
        for epoch in range(1, args.epochs + 1):
    
            train_acc, train_loss = baseline_train(model, trainloader, optimizer, epoch)
            val_acc, val_loss = baseline_validation(model, valloader, optimizer, epoch)
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())
            val_accuracies.append(val_acc.item())
            # Save
            if val_loss < best_loss:  # TODO: maybe use accuracy (not sure if accuracy is a good measurement)
                best_loss = val_loss
                best_acc = max(val_acc, best_acc)
                best_epoch = epoch
            if epoch - best_epoch >= 5:
                # update learning rate
                optimizer.increase_delta()
                best_epoch = epoch
        save_model_state(args.out_path, args.epochs,'model'+str(model_i), model, optimizer,
                         [train_accuracies, val_accuracies], [train_losses, val_losses])

def save_model_state(output_path, epoch, model_name='', model=None, optimizer=None, accuracies=None, losses=None, full=False):
    if full:
        print("Saving full model...")
        name = model_name + '_model_full.pt'
        torch.save(model, os.path.join(output_path, name))
    else:
        print("saving model at epoch:", epoch)
        if not (model is None and optimizer is None):
            name = model_name + '_modelstate_epoch' + str(epoch) + '.pt'
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
