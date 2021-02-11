import argparse
import datetime
import os
import pickle
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from util import ecg_datasets2


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
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/14_01_21-14',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/14_01_21-15',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/14_01_21-11',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_01_21-19',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_01_21-18',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_01_21-16',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_01_21-15',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_21-19',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_21-18',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_21-17',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_21-15',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/03_01_21-19',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/14_01_21-16',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/18_01_21-14',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/19_01_21-18',
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/19_01_21-14'
    ]
    #infer class from model-arch file
    for m_f in model_folders:
        with open(os.path.join(m_f, 'model-arch.txt'), 'r') as arch_file:
            l1 = arch_file.readline(1)
            model_class = eval(l1.strip())
            model_class() #put params here?? or retrain
    loaders = [
        trainloader,
        valloader
    ]
    metric_functions = [ #Functions that take two tensors as argument and give score or list of score

    ]
    for model_i, model_path in enumerate(models):
        #torch.load(model, os.path.join(args.out_path, str(i)+'_model_full.pt'))
        #load model here into model
        model = nn.Linear(1,1)
        model.cuda()
        for dataloader_i, dataloader in enumerate(loaders):
            metrics = [[]*len(metric_functions)]
            for dataset_tuple in dataloader:
                if len(dataset_tuple) == 2:
                    data, labels = dataset_tuple
                elif len(dataset_tuple) == 3:
                    data, labels, _ = dataset_tuple
                else:
                    print("Unknown dataset return tuple length")
                    return
                pred = model(data, y=None).cpu()
                for i, fn in enumerate(metric_functions):
                    metrics[i].append(fn(labels, pred))
            pickle_name = "model-{}-dataset-{}.pickle".format('-'.join(model_path.split(os.sep)), dataloader_i)
            with open(os.path.join(args.out_path, pickle_name), 'wb') as pick_file:
                pickle.dump(metrics, pick_file)
            print("\tFinished dataset {}. Progress: {}/{}".format(dataloader_i, dataloader_i+1, len(loaders)))
        print("Finished model {}. Progress: {}/{}".format(model_path, model_i+1, len(models)))


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
