import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import glob
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def zero_fit_score(labels, predictions, average:str=None):
    #print(labels.shape) #== samples x n_classes
    _, n_classes = labels.shape
    labels = labels.astype(float)
    mask = labels == 0.0
    dists = np.square(np.where(mask, labels - predictions, 0.0))
    if average == 'micro': #Average over all samples
        return 1.0 - np.sum(dists) / np.sum(mask)  # zero fit goal
    if average == 'macro':
        class_average = np.sum(dists, axis=0)/np.sum(mask, axis=0)
        return 1.0 - np.nanmean(class_average)
    else: #Dont take average
        return 1.0 - np.sum(dists, axis=0)/np.sum(mask, axis=0)


def class_fit_score(labels, predictions, average:str=None) -> float:
    # print(labels.shape) #== samples x n_classes
    _, n_classes = labels.shape
    labels = labels.astype(float)
    mask = labels != 0.0
    dists = np.sqrt(np.square(np.where(mask, labels - predictions, 0.0)))
    if average == 'micro':  # Average over all samples
        return 1.0 - np.sum(dists) / np.sum(mask)  # zero fit goal
    if average == 'macro':
        class_average = np.sum(dists, axis=0) / np.sum(mask, axis=0)
        return 1.0 - np.nanmean(class_average)
    else:  # Dont take average
        return 1.0 - np.sum(dists, axis=0) / np.sum(mask, axis=0)

def top1_score(labels, predictions):
    correct = 0
    total = 0
    for l_i in range(len(labels)):
        label = labels[l_i]
        pred = predictions[l_i]
        unique_label_probs = list(reversed(sorted(set(label)-{0.0})))
        pred_prob_idxs = np.argsort(pred)[::-1]
        total_local = 0
        for prob in unique_label_probs:
            label_prob_idxs = np.argwhere(label == prob) #Get idxs of all occurences
            for i in range(len(label_prob_idxs)):
                correct += (pred_prob_idxs[i+total_local] in label_prob_idxs) and pred[pred_prob_idxs[i+total_local]]>0 #add total
            total_local += len(label_prob_idxs)
        total += total_local
    return 1.0*correct/total

def ROC(labels:np.ndarray, predictions:np.ndarray): #see scikit learn doc
    n_classes = labels.shape[1]
    fpr = dict()
    tpr = dict()
    thresholds = dict()  # no dict because always the same
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return tpr, fpr, roc_auc, thresholds

def precision_recall(labels:np.ndarray, predictions:np.ndarray, n_classes): #see scikit learn doc
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes): # For each class
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            predictions[:, i])
        average_precision[i] = average_precision_score(labels[:, i], predictions[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(),
                                                                    predictions.ravel())
    average_precision["micro"] = average_precision_score(labels, predictions,
                                                         average="micro")
    average_precision["macro"] = average_precision_score(labels, predictions,
                                                         average="macro")
    print('Average precision score, micro-averaged over all classes: {0:0.5f}'
          .format(average_precision["micro"]))
    print('Average precision score, macro-averaged over all classes: {0:0.5f}'
          .format(average_precision["macro"]))
    return precision, recall, average_precision

def f1_scores_with_class_counts(counts):
    f_score = dict()
    for i in range(len(counts)):
        Nii = counts[i, i]
        NiX = np.sum(counts[i, :])
        NXi = np.sum(counts[:, i])
        f_score[i] = 2*Nii/(NiX+NXi) if (NiX+NXi)!=0 else 0
    return f_score

def brier_score(labels, predictions, average=None):
    print(labels.shape)
    labels = labels.astype(float)
    dists = np.square(labels - predictions)
    if average == 'micro':  # Average over all samples
        return 1.0 - np.nanmean(dists)  # zero fit goal
    if average == 'macro':
        class_average = np.nanmean(dists, axis=0)
        return 1.0 - np.nanmean(class_average)
    else:  # Dont take average
        return 1.0 - np.nanmean(dists, axis=0)



def select_best_thresholds(tpr, fpr, thresholds, n_classes):
    best_threshold = dict()
    tps = dict()
    fps = dict()
    for i in range(n_classes):
        if np.isnan(fpr[i]).all() or np.isnan(tpr[i]).all():
            best_threshold[i]=np.nan
        else:
            ix = np.argmax(tpr[i]-fpr[i]) #youden J
            #ix = np.argmax(np.sqrt(tpr[i] * (1-fpr[i]))) #gmeans
            best_threshold[i] = thresholds[i][ix]
            tps[i] = tpr[i][ix]
            fps[i] = fpr[i][ix]
    return tps, fps, best_threshold

def class_count_table(labels:np.ndarray, binary_predictions:np.ndarray, n_classes=94):
    counts = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            counts[i, j] = np.sum(labels[:, i] == binary_predictions[:, j])
    return counts

def convert_pred_to_binary(predictions, thresholds):
    thresholds_np = np.array([t for t in thresholds.values()])
    return predictions >= thresholds_np

def read_output_csv_from_model_folder(model_folder = 'models/10_03_21-18/architectures_cpc.cpc_combined.CPCCombined1'):
    pred_path = glob.glob(os.path.join(model_folder, '*output.csv'))[0]
    dfp = pd.read_csv(pred_path)
    return dfp.values[:, 1:].astype(float), dfp.columns[1:].values #1 is file

def read_label_csv_from_model_folder(model_folder):
    label_path = glob.glob(os.path.join(model_folder, '*labels*.csv'))[0]
    dfl = pd.read_csv(label_path)
    labels = dfl.values[:, 1:].astype(float)
    return labels, dfl.columns[1:].values

def read_binary_label_csv_from_model_folder(model_folder):
    label_path = glob.glob(os.path.join(model_folder, '*labels*.csv'))[0]
    dfl = pd.read_csv(label_path)
    labels = dfl.values[:, 1:].astype(int) #convert to int for binary pred
    return labels, dfl.columns[1:].values

if __name__ == '__main__': #usage example
    model_folder = '/home/julian/Downloads/Github/contrastive-predictive-coding/models/22_04_21-17/architectures_cpc.cpc_combined.CPCCombined0'
    labels, _= read_binary_label_csv_from_model_folder(model_folder)
    pred, _ = read_output_csv_from_model_folder(model_folder)
    n_classes = pred.shape[1]
    tpr, fpr, roc_auc, thresholds = ROC(labels, pred)
    tps, fps, best_thresholds = select_best_thresholds(tpr, fpr, thresholds, n_classes)
    binary_preds = convert_pred_to_binary(pred, best_thresholds)
    precision, recall, avg_precision = precision_recall(labels, pred, n_classes)
    counts = class_count_table(labels, binary_preds)
    scores = f1_scores_with_class_counts(counts)
    import util.visualize.plot_metrics as plotm
    plotm.plot_roc_multiclass(tpr, fpr, roc_auc, n_classes)
    plotm.plot_roc_singleclass(tpr, fpr, roc_auc, 59)
    plotm.plot_precision_recall_multiclass(precision, recall, avg_precision, n_classes)
