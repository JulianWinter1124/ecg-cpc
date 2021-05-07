import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
from sklearn.metrics import auc


def plot_roc_singleclass(tpr, fpr, roc_auc, class_name, class_i, savepath=None, plot_name=''):
    plt.figure()
    lw = 2
    plt.plot(fpr[class_i], tpr[class_i], color='darkorange',
             lw=lw, label='ROC curve: ' + class_name + '(area = %0.2f)' % roc_auc[class_i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name+'\nReceiver operating characteristic for single class: ' + class_name)
    plt.legend(loc="lower right")
    if savepath:
        plt.savefig(os.path.join(savepath, f'roc-{class_name}.png'), bbox_inches='tight')
    plt.show()

def plot_precision_recall_microavg(recall, precision, average_precision, savepath=None):
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    if savepath:
        plt.savefig(os.path.join(savepath, 'precision-recall-microavg.png'))


def plot_precision_recall_multiclass(precision, recall, average_precision, classes, selection=None, savepath=None, plot_name=''):
    n_classes = len(classes)
    cm = plt.get_cmap('gist_rainbow')
    if selection is None:
        selection = range(n_classes)
    # setup plot details
    colors = cycle([cm(1.*i/n_classes) for i in selection])

    plt.figure(figsize=(12, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    lines.append(l) #yes only one
    labels.append('iso-f1 curves')

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(selection, colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.2f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(plot_name+'\nExtension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    if savepath:
        plt.savefig(os.path.join(savepath, 'precision-recall-multiclass.png'), bbox_inches='tight')
    plt.show()

def plot_roc_multiclass(tpr, fpr, roc_auc, classes:list, selection=None, savepath=None, plot_name=''):
    n_classes = len(classes)
    selection = range(n_classes) if selection is None else selection
    all_fpr = np.unique(np.concatenate([fpr[i] for i in selection]))
    lw = 1
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in selection:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(selection, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name+'\nReceiver operating characteristic for multi-class')
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=6)
    if savepath:
        plt.savefig(os.path.join(savepath, 'ROC-multiclass.png'), bbox_inches='tight')
    plt.show()