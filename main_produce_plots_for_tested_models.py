import glob
import os

import torch

from util.metrics.baseline_losses import bidirectional_cross_entropy, cross_entropy, binary_cross_entropy
from util.metrics import metrics as m
import util.visualize.plot_metrics as plotm

def auto_find_tested_models(path='models/'):
    #only works with specific file structure date/modelfolder/files
    csvs = glob.glob(os.path.join(path, '*/*/*.csv')) #Finds all csv files with above structure
    csv_paths = list(reversed(list(set([os.path.abspath(os.path.split(csv)[0]) for csv in csvs]))))
    return csv_paths

def create_metric_plots_1(model_folder, labels, pred):
    print("creating for:", model_folder)
    n_classes = pred.shape[1]
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
    counts = m.class_count_table(labels, binary_preds)
    scores = m.f1_scores_with_class_counts(counts)

    plotm.plot_roc_multiclass(tpr, fpr, roc_auc, n_classes)
    plotm.plot_roc_singleclass(tpr, fpr, roc_auc, 59)
    plotm.plot_precision_recall_multiclass(precision, recall, avg_precision, n_classes)

if __name__ == '__main__':
    model_folders = ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/27_04_21-18/architectures_baseline_challenge.baseline_cnn_v9.BaselineNet0']#auto_find_tested_models() #or manual list
    print(model_folders)
    for model_folder in model_folders:
        try:
            labels, classes = m.read_binary_label_csv_from_model_folder(model_folder)
            pred, pred_classes = m.read_output_csv_from_model_folder(model_folder)
            create_metric_plots_1(model_folder, labels, pred)
        except Exception as e: #folder with not the correct csv?
            print(e)