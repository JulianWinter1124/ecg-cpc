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
from torch.utils.data import DataLoader

from architectures_baseline import baseline_losses, baseline_cnn_v0, baseline_cnn_v2, baseline_cnn_v3, \
    baseline_cnn_v4
import accuracy_metrics
from util.data import ecg_datasets2
from util.full_class_name import fullname



def main(args):
    np.random.seed(args.seed)
    print(args.out_path)
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
        '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        window_size=9500)
    print(args.test_mode) #TODO: is always True for some reason
    args.test_mode = False
    model_classes = [
        baseline_cnn_v0.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        baseline_cnn_v2.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        baseline_cnn_v3.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False),
        baseline_cnn_v4.BaselineNet(in_channels=args.channels, out_channels=args.latent_size, out_classes=args.forward_classes, verbose=False)
    ]

    loaders = [
        DataLoader(train_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score
        accuracy_metrics.accuracy,
        accuracy_metrics.fn_score_label,
        accuracy_metrics.tn_score_label,
        accuracy_metrics.tp_score_label,
        accuracy_metrics.f1_score
    ]
    for model_i, model in enumerate(model_classes):
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name)
        #Create dirs and model info
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'params.txt'), 'w') as cfg:
            cfg.write(str(args))
        with open(os.path.join(output_path, 'model_arch.txt'), 'w') as f:
            print(fullname(model), file=f)
            print(model, file=f)
        model.cuda()
        model.train()
        #init optimizer
        optimizer = Adam(model.parameters(), lr=1e-3)
        for dataloader_i, dataloader in enumerate(loaders):
            metrics = defaultdict(lambda: defaultdict(list))
            for epoch in range(1, args.epochs+1):
                starttime = time.time()
                for dataset_tuple in dataloader:
                    data, labels = dataset_tuple
                    data = data.float().cuda()
                    labels = labels.float().cuda()
                    optimizer.zero_grad()
                    pred = model(data, y=None) #makes model return prediction instead of loss
                    loss = baseline_losses.MSE_loss(pred=pred, y=labels)
                    loss.backward()
                    optimizer.step()
                    #saving metrics
                    metrics[epoch]['loss'].append(parse_tensor_to_numpy_or_scalar(loss))
                    for i, fn in enumerate(metric_functions):
                        metrics[epoch]['acc_'+str(i)].append(parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                    if args.test_mode:
                        break
                elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
                metrics[epoch]['elapsed_time'].append(elapsed_time)
                print("Epoch {}/{} done. Avg loss: {:.4f} Elapsed time: {}".format(
                    epoch, args.epochs, np.mean(metrics[epoch]['loss']), elapsed_time))
                if args.test_mode:
                    break
            print("\tFinished dataset {}. Progress: {}/{}".format(dataloader_i, dataloader_i + 1, len(loaders)))
            pickle_name = "model-{}-dataset-{}-epochs-{}.pickle".format(model_name, dataloader_i, args.epochs)
            #Saving metrics in pickle
            with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
                pickle.dump(dict(metrics), pick_file)
            if args.test_mode:
                break
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i+1, len(model_classes)))
        #TODO: Save model + model weights + optimizer state

        #delete and free
        del model
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

    parser.add_argument('--test_mode', type=bool, default=False, action='store_false',
                        help="Only run minimal samples to test all models functionality")

    args = parser.parse_args()
    main(args)
