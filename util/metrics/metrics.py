import numpy as np
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, roc_auc_score
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

def confusion_matrix(binary_labels, binary_predictions):
    return np.array(multilabel_confusion_matrix(binary_labels, binary_predictions), dtype=int)


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

def auc_scores(labels, predictions, average=None):
    auc = roc_auc_score(labels, predictions, average=average)
    return auc


def precision_scores(binary_labels, binary_predictions, average=None):
    cm = confusion_matrix(binary_labels, binary_predictions) # n_classes x 2 x 2
    tn = cm[:, 0, 0] #TN
    fn = cm[:, 1, 0] #FN
    fp = cm[:, 0, 1] #FP
    tp = cm[:, 1, 1] #TP
    if average == 'macro':
        return np.mean(tp/(tp+fp))
    elif average == 'micro':
        return np.sum(tp)/(np.sum(tp)+np.sum(fp)) #Micro
    else:
        return tp/(tp+fp)

def recall_scores(binary_labels, binary_predictions, average=None):
    cm = confusion_matrix(binary_labels, binary_predictions) # n_classes x 2 x 2
    tn = cm[:, 0, 0] #TN
    fn = cm[:, 1, 0] #FN
    fp = cm[:, 0, 1] #FP
    tp = cm[:, 1, 1] #TP
    if average == 'macro':
        return np.mean(tp/(tp+fn))
    elif average == 'micro':
        return np.sum(tp)/(np.sum(tp)+np.sum(fn)) #Micro
    else:
        return tp/(tp+fn)

def accuracy_scores(binary_labels, binary_predictions, average=None):
    cm = confusion_matrix(binary_labels, binary_predictions) # n_classes x 2 x 2
    tn = cm[:, 0, 0] #TN
    fn = cm[:, 1, 0] #FN
    fp = cm[:, 0, 1] #FP
    tp = cm[:, 1, 1] #TP
    if average == 'macro':
        return np.mean((tp+tn)/(tp+tn+fp+fn))
    elif average == 'micro':
        return (np.sum(tp)+np.sum(tn))/(np.sum(tp)+np.sum(fp)+np.sum(tn)+np.sum(fn)) #Micro
    else:
        return (tp+tn)/(tp+tn+fp+fn)

def balanced_accuracy_scores(binary_labels, binary_predictions, average=None):
    cm = confusion_matrix(binary_labels, binary_predictions) # n_classes x 2 x 2
    tn = cm[:, 0, 0] #TN
    fn = cm[:, 1, 0] #FN
    fp = cm[:, 0, 1] #FP
    tp = cm[:, 1, 1] #TP
    #TPR = (tp/(tp+fn))
    #TNR = (tn/(tn+fp))
    if average == 'macro':
        return np.mean(((tp/(tp+fn))+(tn/(tn+fp)))/2)
    elif average == 'micro':
        return ((np.sum(tp)/(np.sum(tp)+np.sum(fn)))+(np.sum(tn)/(np.sum(tn)+np.sum(fp))))/2
    else:
        return ((tp/(tp+fn))+(tn/(tn+fp)))/2

def f1_scores(binary_labels, binary_predictions, average=None):
    cm = confusion_matrix(binary_labels, binary_predictions) # n_classes x 2 x 2
    tn = cm[:, 0, 0] #TN
    fn = cm[:, 1, 0] #FN
    fp = cm[:, 0, 1] #FP
    tp = cm[:, 1, 1] #TP
    if average == 'macro':
        return np.mean(tp/(tp+(fp+fn)/2.0))
    elif average == 'micro':
        return np.sum(tp)/(np.sum(tp)+(np.sum(fp)+np.sum(fn))/2.0) #Micro
    else:
        return tp/(tp+(fp+fn)/2.0)



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
    return (predictions >= thresholds_np).astype(int)

def read_output_csv_from_model_folder(model_folder, data_loader_index=0):
    pred_path = glob.glob(os.path.join(model_folder, f"model-*-dataloader-{data_loader_index}-output.csv"))
    if len(pred_path) == 0:
        raise FileNotFoundError
    dfp = pd.read_csv(pred_path[0])
    return dfp.values[:, 1:].astype(float), dfp.columns[1:].values #1 is file

def read_label_csv_from_model_folder(model_folder, data_loader_index=0):
    label_path = glob.glob(os.path.join(model_folder, f"labels-dataloader-{data_loader_index}.csv"))
    if len(label_path) == 0:
        raise FileNotFoundError
    dfl = pd.read_csv(label_path[0])
    labels = dfl.values[:, 1:].astype(float)
    return labels, dfl.columns[1:].values

def read_binary_label_csv_from_model_folder(model_folder, data_loader_index=0):
    l, c = read_label_csv_from_model_folder(model_folder, data_loader_index)
    l = (l > 0).astype(int)
    return l, c

if __name__ == '__main__': #usage example
    model_folder = '/home/julian/Downloads/Github/contrastive-predictive-coding/models/22_04_21-17/architectures_cpc.cpc_combined.CPCCombined0'
    labels, _= read_binary_label_csv_from_model_folder(model_folder)
    print(labels[0])
    pred, _ = read_output_csv_from_model_folder(model_folder)
