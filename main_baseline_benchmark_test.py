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
from torch.utils.data import DataLoader

from architectures_baseline import baseline_losses
from util.metrics import training_metrics
from util.data import ecg_datasets2
from util.full_class_name import fullname
from util.store_models import save_model_checkpoint
from util import store_models



def main(args):
    np.random.seed(args.seed)
    print(args.out_path)
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    train_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
        '/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        window_size=9500)
    val_dataset_ptbxl = ecg_datasets2.ECGDatasetBaselineMulti(
        '/media/julian/data/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/val',
        # '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels/train',
        window_size=9500)
    model_folders = [
        'models/15_02_21-14/architectures_baseline.baseline_cnn_v0.BaselineNet'
    ]

    train_loaders = [
        DataLoader(train_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
    ]
    val_loaders = [
        DataLoader(val_dataset_ptbxl, batch_size=args.batch_size, drop_last=True, num_workers=1, collate_fn=ecg_datasets2.collate_fn)
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score #TODO: maybe use dict with name
        #accuracy_metrics.fn_score_label,
        #accuracy_metrics.tn_score_label,
        #accuracy_metrics.tp_score_label,
        training_metrics.f1_score,
        training_metrics.micro_avg_recall_score,
        training_metrics.micro_avg_precision_score,
        training_metrics.accuracy
    ]
    for model_i, model_path in enumerate(model_folders):
        model_arch_path = glob.glob(os.path.join(model_path, '*full_model.pt'))[0]
        model_checkpoint_path = glob.glob(os.path.join(model_path, '*checkpoint*.pt'))[-1]
        print('Found path to model architecture {} and checkpoint {}'.format(model_arch_path, model_checkpoint_path))
        model = store_models.load_model_architecture(model_arch_path) #load correct class
        optimizer = Adam(model.parameters())
        model, optimizer, epoch = store_models.load_model_checkpoint(model_checkpoint_path, model, optimizer) #load model weights
        model_name = fullname(model)
        output_path = os.path.join(args.out_path, model_name)
        #Create dirs
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.cuda()
        model.train()
        metrics = defaultdict(lambda: defaultdict(list))
        starttime = time.time() #train
        for loader_i, loader in enumerate(val_loaders): #validate
            for dataset_tuple in loader:
                data, labels = dataset_tuple
                data = data.float().cuda()
                labels = labels.float().cuda()
                pred = model(data, y=None) #makes model return prediction instead of loss
                loss = baseline_losses.MSE_loss(pred=pred, y=labels)
                #saving metrics
                if args.save_metrics:
                    metrics[epoch]['loss'].append(parse_tensor_to_numpy_or_scalar(loss))
                    for i, fn in enumerate(metric_functions):
                        metrics[epoch]['val_acc_'+str(i)].append(parse_tensor_to_numpy_or_scalar(fn(y=labels, pred=pred)))
                if args.save_predictions:
                    pass
                if args.dry_run:
                    break
            print("\tFinished dataset {}. Progress: {}/{}".format(loader_i, loader_i + 1, len(val_loaders)))
            del data
            del labels

        elapsed_time = str(datetime.timedelta(seconds=time.time() - starttime))
        metrics[epoch]['elapsed_time'].append(elapsed_time)
        print("Epoch {}/{} done. Avg train loss: {:.4f}. Avg val loss: {:.4f} Elapsed time: {}".format(
            epoch, args.epochs, np.mean(metrics[epoch]['trainloss']), np.mean(metrics[epoch]['valloss']), elapsed_time))
        pickle_name = "model-{}-epochs-{}.pickle".format(model_name, args.epochs)
        #Saving metrics in pickle
        with open(os.path.join(output_path, pickle_name), 'wb') as pick_file:
            pickle.dump(dict(metrics), pick_file)
        #Save model + model weights + optimizer state
        save_model_checkpoint(output_path, epoch=args.epochs, model=model, optimizer=optimizer, name=model_name)
        print("Finished model {}. Progress: {}/{}".format(model_name, model_i+1, len(model_folders)))
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

    parser.add_argument('--save_metrics', dest='save_metrics', action='store_true',
                        help="Pass this argument to save metrics in a pickle")
    parser.set_defaults(save_metrics=False)

    parser.add_argument('--save_predictions', dest='save_predictions', action='store_true',
                        help="Pass this argument to save model output in file")
    parser.set_defaults(save_predictions=False)

    args = parser.parse_args()
    main(args)