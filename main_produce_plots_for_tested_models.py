import glob
import os
import pandas as pd
import torch
import numpy as np

from util.metrics.baseline_losses import bidirectional_cross_entropy, cross_entropy, binary_cross_entropy
from util.metrics import metrics as m
import util.visualize.plot_metrics as plotm

def auto_find_tested_models(path='models/'):
    #only works with specific file structure date/modelfolder/files
    csvs = glob.glob(os.path.join(path, '*/*/*.csv')) #Finds all csv files with above structure
    csv_paths = list(reversed(list(set([os.path.abspath(os.path.split(csv)[0]) for csv in csvs]))))
    return csv_paths

def create_metric_plots_1(model_folder, labels, pred, classes):
    print("creating for:", model_folder)
    n_classes = len(classes)
    tpr, fpr, roc_auc, thresholds = m.ROC(labels, pred)
    tps, fps, best_thresholds = m.select_best_thresholds(tpr, fpr, thresholds, n_classes)
    zero_fit = m.zero_fit_score(labels, pred, 'macro')
    print('zero_fit, macro', zero_fit)
    zero_fit = m.zero_fit_score(labels, pred, 'micro')
    print('zero_fit, micro', zero_fit)
    class_fit = m.class_fit_score(labels, pred, 'macro')
    print('class_fit, macro', class_fit)
    class_fit = m.class_fit_score(labels, pred, 'micro')
    print('class_fit, micro', class_fit)
    binary_preds = m.convert_pred_to_binary(pred, best_thresholds)
    precision, recall, avg_precision = m.precision_recall(labels, pred, n_classes)
    counts = m.class_count_table(labels, binary_preds, n_classes)
    scores = m.f1_scores_with_class_counts(counts)
    print('macro', m.brier_score(labels, pred, 'macro'))
    print('micro', m.brier_score(labels, pred, 'micro'))
    class_brier_scores = m.brier_score(labels, pred)

    normal_class = '426783006'
    normal_class_idx = np.where(classes == normal_class)[0][0]
    plotm.plot_roc_multiclass(tpr, fpr, roc_auc, classes)
    plotm.plot_roc_singleclass(tpr, fpr, roc_auc, class_name=normal_class, class_i=normal_class_idx)
    plotm.plot_precision_recall_multiclass(precision, recall, avg_precision, classes)

def create_csv_table(output_folder, filename, labels, pred, classes):
    n_classes = len(classes)


def save_csv(output_folder, filename, data, column_titles):
    df = pd.DataFrame(data, columns=column_titles)
    p = os.path.join(output_folder, filename)
    df.to_csv(p)



if __name__ == '__main__':
    model_folders = ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/30_04_21-20/architectures_baseline_challenge.baseline_cnn_v6.BaselineNet0']#auto_find_tested_models() #or manual list
    print(model_folders)
    for model_folder in model_folders:
        try:
            labels, classes = m.read_binary_label_csv_from_model_folder(model_folder)
            print(len(classes), classes)
            pred, pred_classes = m.read_output_csv_from_model_folder(model_folder)
            create_metric_plots_1(model_folder, labels, pred, classes)
        except FileNotFoundError as e: #folder with not the correct csv?
            print(e)