import os
from itertools import cycle

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn
from matplotlib.font_manager import FontProperties
from numpy import interp
from sklearn.metrics import auc, ConfusionMatrixDisplay


def plot_roc_singleclass(tpr, fpr, roc_auc, class_name, class_i, savepath=None, plot_name='', plot_legends=True):
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
    if plot_legends:
        plt.legend(loc="lower right")
    if savepath:
        plt.savefig(os.path.join(savepath, f'roc-{class_name}.png'), bbox_inches='tight')
    plt.show()

def plot_roc_multiclass(tpr, fpr, roc_auc, classes:list, selection=None, savepath=None, plot_name='', plot_legends=True):
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

    cm = plt.get_cmap('gist_rainbow')
    colors = cycle([cm(1.*i/n_classes) for i in selection])

    for i, color in zip(selection, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    parts = plot_name.split('|')
    plot_name = parts[0]+'\n'+'|'.join(parts[1:])
    plt.title(plot_name+'\nReceiver operating characteristic for multi-class')
    if plot_legends:
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=6)
    if savepath:
        plt.savefig(os.path.join(savepath, 'ROC-multiclass.png'), bbox_inches='tight')
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


def plot_precision_recall_multiclass(precision, recall, average_precision, classes, selection=None, savepath=None, plot_name='', plot_legends=True):
    n_classes = len(classes)
    if selection is None:
        selection = range(n_classes)
    # setup plot details
    cm = plt.get_cmap('gist_rainbow')
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
    parts = plot_name.split('|')
    plot_name = parts[0]+'\n'+'|'.join(parts[1:])
    plt.title(plot_name+'\nExtension of Precision-Recall curve to multi-class')

    if plot_legends:
        plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    if savepath:
        plt.savefig(os.path.join(savepath, 'precision-recall-multiclass.png'), bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(confusion_matrix:np.ndarray, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    disp.plot()

def plot_parallel_coordinates(df: pd.DataFrame, color_column, save_to, drop_columns:list=[], put_last_columns:list=[], exclude_color_column=True):
    def _make_plotly_dict(column_name, data):
        d = dict()
        t = data.dtype
        if t == bool:
            d['range'] = [-0.5,1.5]
            d['tickvals'] = [True, False]
            d['ticktext'] = ['True', 'False']
            d['values'] = data
        elif t == str or t == object:
            da = data.astype('category').cat
            d['tickvals'] = da.codes
            d['ticktext'] = da.categories
            d['values'] = da.codes
        elif t == int:
            d['range'] = [data.min(), data.max()]
            d['tickformat'] = 'd'
            d['values'] = data
        else:
            d['range'] = [data.min(), data.max()]
            d['values'] = data
        d['label'] = column_name
        return d
    cols = df.columns.tolist()
    #cols = list(set(cols)-set(put_last_columns)-set(exclude_columns))+put_last_columns
    cols = [c for c in cols if not (c in drop_columns or c in put_last_columns)] + put_last_columns
    df = df[cols]
    dimensions = [_make_plotly_dict(column_name, data) for column_name, data in df.iteritems() if not (exclude_color_column and column_name == color_column)]
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df[color_column],
                       colorscale = 'Electric',
                        #autocolorscale=True,
                       showscale = True,
                       cmin = df[color_column].min(),
                       cmax = df[color_column].max()),
            dimensions = dimensions
        )
    )
    fig.update_traces(labelangle=-90, selector=dict(type='parcoords'))
    fig.show()
    if not save_to is None:
        fig.write_image(save_to)


def plot_lowlabel_availabilty(df_groups, title, save_to, filename, data_col='micro', save_legend_seperate=False):
    fractions = {'fewer-labels0_001': 0.0012756005952802778,
                 'fewer-labels0_005': 0.009378026598634634,
                 'fewer-labels0_05': 0.10032362459546926,
                 'fewer-labels0_01': 0.02010252049228734,
                 'fewer-labels10': 0.18834006566980843,
                 'fewer-labels14': 0.18829282120331656,
                 'fewer-labels20': 0.3308057543760187,
                 'fewer-labels30': 0.44031842770415514,
                 'fewer-labels40': 0.5257600453546878,
                 'fewer-labels50': 0.5950912999314956,
                 'fewer-labels60': 0.6526823045850755,
                 'train-test-splits': 0.7017220608036284,
                 'train-test-splits_min_cut10': 0.015330829376609265,
                 'train-test-splits_min_cut25': 0.037252261828833295,
                 'train-test-splits_min_cut50': 0.06798478728178962,
                 'train-test-splits_min_cut100': 0.116174143103489,
                 'train-test-splits_min_cut150': 0.15470200552760258,
                 'train-test-splits_min_cut200': 0.18864715470200552}
    ordered_splits = [k for k, v in sorted(fractions.items(), key=lambda item: item[1])]
    def get_fraction_x_for_splitsname(name):
        return fractions[name]
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(figsize=(20,10))
    seaborn.set_palette("hls", len(df_groups))
    plt.title(title)
    for name, group in df_groups:
        g = group.reset_index()
        g.insert(loc=0, column='splitfraction', value=[get_fraction_x_for_splitsname(sp) for sp in g['level_0']])
        g = g.sort_values(by='splitfraction')
        if 'cpc' in name.lower():
            ax.plot(g['splitfraction'], g[data_col], '--o', label=name)
        else:
            ax.plot(g['splitfraction'], g[data_col], '-o', label=name)
    plt.ylabel('average AUC score')

    plt.xlabel('fraction of files used (i.r.t. all files)')
    handles, labels = ax.get_legend_handles_labels()
    if save_legend_seperate:
        legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
        export_legend(legend, save_to, 'legend-'+filename)
    else:
        ax.legend(handles, labels, loc='lower right', prop=fontP, handlelength=3) #bbox_to_anchor=(1.05, 1)
    fig.savefig(os.path.join(save_to, filename), bbox_inches='tight')
    plt.show()

def export_legend(legend, save_to, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_to, filename), dpi="figure", bbox_inches=bbox)
