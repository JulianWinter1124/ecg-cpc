import glob
import json
import os
import pathlib
import re
from datetime import datetime

import pandas as pd
import numpy as np

from util.data.dataframe_factory import DataFrameFactory
from util.metrics import metrics as m
import util.visualize.plot_metrics as plotm

from util.utility.dict_utils import count_key_in_dict, extract_values_for_key_in_dict


def auto_find_tested_models_recursive(path='models/'):
    # only works with specific file structure date/modelfolder/files
    print(f"Looking for models at {os.path.abspath(path)}")
    files = []
    for root, dirs, dir_files in os.walk(path):
        fm_temp, ch_temp = [], []
        for file in dir_files:
            # print('Checking', file, 'labels' in file and file.endswith('.csv'), 'output' in file and file.endswith('.csv'))
            if 'labels' in file and file.endswith('.csv'):
                fm_temp.append(os.path.join(root, file))
            if 'output' in file and file.endswith('.csv'):
                ch_temp.append(os.path.join(root, file))
        if len(fm_temp) > 0 and len(ch_temp) > 0:
            files.append(os.path.split(fm_temp[0])[0])
        else:
            print(f"not a model folder: {root}")
    print(f"Found {len(files)} model test files")
    return files


def auto_find_tested_models(path='models/'):
    csvs = glob.glob(os.path.join(path, '*/*/*.csv'))  # Finds all csv files with above structure
    csv_paths = list(reversed(list(set([os.path.abspath(os.path.split(csv)[0]) for csv in csvs]))))
    return csv_paths


def long_to_shortname(model_name):
    print('modelname in ', model_name)
    model_name = re.sub('cpc_combined.CPCCombined', 'CPC:', model_name)
    model_name = re.sub('architectures_baseline_challenge.', '', model_name)
    model_name = re.sub('baseline_cnn', 'BL', model_name)
    model_name = re.sub('baseline', 'BL', model_name)
    model_name = re.sub('.BaselineNet', ':', model_name)
    model_name = re.sub('cpc_downstream_only', 'linear', model_name)
    model_name = re.sub('cpc_downstream_', '', model_name)
    # model_name = re.sub('|use_weights', '', model_name)
    # model_name = re.sub('|pte:\d*', '', model_name)
    return model_name


# def create_metric_plots(model_folder, binary_labels, pred, classes):
#     print("creating for:", model_folder)
#     model_folder_name = os.path.split(model_folder)[1]
#     n_classes = len(classes)
#     tpr, fpr, roc_auc, thresholds = m.ROC(binary_labels, pred)
#     tps, fps, best_thresholds = m.select_best_thresholds(tpr, fpr, thresholds, n_classes)
#     zero_fit = m.zero_fit_score(binary_labels, pred, 'macro')
#     print('zero_fit, macro', zero_fit)
#     zero_fit = m.zero_fit_score(binary_labels, pred, 'micro')
#     print('zero_fit, micro', zero_fit)
#     class_fit = m.class_fit_score(binary_labels, pred, 'macro')
#     print('class_fit, macro', class_fit)
#     class_fit = m.class_fit_score(binary_labels, pred, 'micro')
#
#     print('class_fit, micro', class_fit)
#     binary_preds = m.convert_pred_to_binary(pred, best_thresholds)
#
#     df = create_metric_score_dataframe(binary_labels, binary_preds, classes, m.f1_scores)
#     print(df)
#     df = create_metric_score_dataframe(binary_labels, binary_preds, classes, m.recall_scores)
#     print(df)
#     df = create_metric_score_dataframe(binary_labels, binary_preds, classes, m.precision_scores)
#     print(df)
#     df = create_metric_score_dataframe(binary_labels, binary_preds, classes, m.accuracy_scores)
#     print(df)
#     df = create_metric_score_dataframe(binary_labels, binary_preds, classes, m.balanced_accuracy_scores)
#     print(df)
#     #print(df.to_latex(index=False, label='', caption='LUL'))
#
#     normal_class = '426783006'
#     normal_class_idx = np.where(classes == normal_class)[0][0]
#     plotm.plot_roc_multiclass(tpr, fpr, roc_auc, classes, savepath=model_folder, plot_name=model_folder_name)
#     plotm.plot_roc_singleclass(tpr, fpr, roc_auc, class_name=normal_class, class_i=normal_class_idx, savepath=model_folder, plot_name=model_folder_name)
#     #plotm.plot_precision_recall_multiclass(precision, recall, avg_precision, classes, savepath=model_folder, plot_name=model_folder_name)

def create_metric_score_dataframe(binary_labels, binary_preds, classes, metric_function, model_name=None,
                                  average_only=False, model_attrs=None):
    scdf = pd.DataFrame({'model':[model_name]})
    if not model_attrs is None:
        scdf = pd.concat([scdf, pd.DataFrame({k:v if type(v)==list else [v] for k,v in model_attrs.items()})], axis=1)
    if not average_only:
        scores = metric_function(binary_labels, binary_preds, average=None)
        scdf = pd.concat([scdf, pd.DataFrame(scores[np.newaxis, :], columns=classes)], axis=1, )
    avgs = pd.DataFrame(data={
        'micro': np.atleast_1d(metric_function(binary_labels, binary_preds, average='micro')),
        'macro': np.atleast_1d(metric_function(binary_labels, binary_preds, average='macro'))
    })
    scdf = pd.concat([scdf, avgs], axis=1)
    # scdf.insert(0, 'micro', metric_function(binary_labels, binary_preds, average='micro'), allow_duplicates=True)
    # scdf.insert(0, 'macro', metric_function(binary_labels, binary_preds, average='macro'), allow_duplicates=True)
    #scdf = scdf.set_index('model')
    return scdf


def create_metric_confusion_matrix(model_folder, binary_labels, pred, classes: list):
    print("creating for:", model_folder)
    model_folder_name = os.path.split(model_folder)[1]
    n_classes = len(classes)
    tpr, fpr, roc_auc, thresholds = m.ROC(binary_labels, pred)
    tps, fps, best_thresholds = m.select_best_thresholds(tpr, fpr, thresholds, n_classes)
    # test_thresholds = {c:0.5 for c in classes}
    binary_preds = m.convert_pred_to_binary(pred, best_thresholds)
    print(binary_preds)
    print(binary_labels)
    cm = m.confusion_matrix(binary_labels, binary_preds)
    print(cm)
    plotm.plot_confusion_matrix(cm, classes)


# def create_latex_table(dataframe:pd.DataFrame, output_folder, latex_label='', caption='', filename='table.tex'):
#     latex_string = dataframe.to_latex(index=False, label=latex_label, caption=caption)
#     with open(os.path.join(output_folder, filename), 'w') as f:
#         f.write(latex_string)


def calculate_best_thresholds(model_folders, data_loader_index=1):  # 0 = test, 1 = val, 2 = train
    model_thresholds = []
    for mi, model_folder in enumerate(model_folders):
        print(model_folder)
        best_thresholds = None
        try:
            binary_labels, classes = m.read_binary_label_csv_from_model_folder(model_folder,
                                                                               data_loader_index=data_loader_index)
            predictions, pred_classes = m.read_output_csv_from_model_folder(model_folder,
                                                                            data_loader_index=data_loader_index)
            if np.any(np.isnan(predictions)):
                print(f"Encountered nan value in {model_folder} prediction!")
                model_thresholds.append(None)
                continue
            tprs, fprs, roc_auc, thresholds = m.ROC(binary_labels, predictions)
            tpr, fpr, best_thresholds = m.select_best_thresholds(tprs, fprs, thresholds, len(classes))
        except FileNotFoundError as e:  # folder with not the correct csv?
            print(e)
        model_thresholds.append(best_thresholds)
    return model_thresholds


def create_paper_plots(model_folders, data_loader_index=0):
    TEST_SET = 0;
    VAL_SET = 1;
    TRAIN_SET = 2

    model_thresholds = calculate_best_thresholds(model_folders, data_loader_index=VAL_SET)
    for mi, model_folder in enumerate(model_folders):
        if model_thresholds[mi] is None:
            print(f"Encountered nan value in {model_folder} prediction!. Skipping this model.")
            continue
        try:
            model_name = os.path.split(model_folder)[1]
            model_name = '.'.join(model_name.split('.')[
                                  -2:]) if '.' in model_name else model_name  # fullname(store_models.load_model_architecture(extract_model_files_from_dir(model_folder)[0][0]))
            print(model_name)
            binary_labels, classes = m.read_binary_label_csv_from_model_folder(model_folder,
                                                                               data_loader_index=data_loader_index)
            predictions, pred_classes = m.read_output_csv_from_model_folder(model_folder,
                                                                            data_loader_index=data_loader_index)
            binary_predictions = m.convert_pred_to_binary(predictions, model_thresholds[mi])

            n_classes = len(classes)
            tpr, fpr, roc_auc, thresholds = m.ROC(binary_labels, predictions)
            tps, fps, best_thresholds = m.select_best_thresholds(tpr, fpr, thresholds, n_classes)
            normal_class = '426783006'
            normal_class_idx = np.where(classes == normal_class)[0][0]
            plotm.plot_roc_multiclass(tpr, fpr, roc_auc, classes, savepath=model_folder, plot_name=model_name,
                                      plot_legends=False)
            plotm.plot_roc_singleclass(tpr, fpr, roc_auc, class_name=normal_class, class_i=normal_class_idx,
                                       savepath=model_folder, plot_name=model_name)
        except FileNotFoundError:
            print("File not found")


def create_paper_metrics(model_folders, root_path, data_loader_index=0, average_only=False, long_tables=False, cut_name_attrs=False, use_attrs_in_name=False, include_attrs_in_table=False,
                         save_to_all_dirs=True):
    TEST_SET = 0;
    VAL_SET = 1;
    TRAIN_SET = 2

    model_thresholds = calculate_best_thresholds(model_folders, data_loader_index=VAL_SET)

    f1_dff = DataFrameFactory()
    prec_dff = DataFrameFactory()
    rec_dff = DataFrameFactory()
    classfit_dff = DataFrameFactory()
    zerofit_dff = DataFrameFactory()
    auc_dff = DataFrameFactory()

    for mi, model_folder in enumerate(model_folders):
        try:
            if model_thresholds[mi] is None:
                print(f"Encountered nan value in {model_folder} prediction!. Skipping this model.")
                continue
            model_name = os.path.split(model_folder)[1]
            model_name = '.'.join(model_name.split('.')[
                                  -2:]) if '.' in model_name else model_name  # fullname(store_models.load_model_architecture(extract_model_files_from_dir(model_folder)[0][0]))
            if cut_name_attrs:
                model_name = model_name.split('|')[0]
            model_name = long_to_shortname(model_name)
            print('Final name:', model_name)
            attrs = None
            if use_attrs_in_name or include_attrs_in_table:
                attrs = extract_model_attributes(model_folder, model_name, False, False)
                name_attrs = model_name
                if use_attrs_in_name:
                    for k, v in attrs.items():
                        if (type(v) == str) and (not v in name_attrs):
                            model_name += '|'+v
                if not include_attrs_in_table:
                    attrs = None
            with open(os.path.join(model_folder, 'thresholds.txt'), 'w+') as f:
                f.write(','.join(map(str, model_thresholds[mi].values())))
            binary_labels, classes = m.read_binary_label_csv_from_model_folder(model_folder,
                                                                               data_loader_index=data_loader_index)
            predictions, pred_classes = m.read_output_csv_from_model_folder(model_folder,
                                                                            data_loader_index=data_loader_index)
            binary_predictions = m.convert_pred_to_binary(predictions, model_thresholds[mi])
            print(model_thresholds[mi])
            # scores with binary
            f1_dff.append(
                create_metric_score_dataframe(binary_labels, binary_predictions, classes, m.f1_scores, model_name,
                                              average_only, model_attrs=attrs))

            prec_dff.append(
                create_metric_score_dataframe(binary_labels, binary_predictions, classes, m.precision_scores,
                                              model_name, average_only, model_attrs=attrs))
            rec_dff.append(
                create_metric_score_dataframe(binary_labels, binary_predictions, classes, m.recall_scores, model_name,
                                              average_only, model_attrs=attrs))
            # scores with probability
            classfit_dff.append(
                create_metric_score_dataframe(binary_labels, predictions, classes, m.class_fit_score, model_name,
                                              average_only, model_attrs=attrs))
            zerofit_dff.append(
                create_metric_score_dataframe(binary_labels, predictions, classes, m.zero_fit_score, model_name,
                                              average_only, model_attrs=attrs))
            auc_dff.append(create_metric_score_dataframe(binary_labels, predictions, classes, m.auc_scores, model_name,
                                                         average_only, model_attrs=attrs))

        except FileNotFoundError as e:  # folder with not the correct csv?
            print(e)
    all_factories = [auc_dff, prec_dff, rec_dff, zerofit_dff, f1_dff, classfit_dff]
    list(map(lambda obj: obj.natsort_by_column(column='model'), all_factories)) #SINCE WHEN ARE THESE LAZY?!
    list(map(lambda obj: obj.put_columns_last(columns=['micro', 'macro']), all_factories))
    auc_dff.to_csv(root_path, 'All_attributes_with_scores.csv')
    try:
        list(map(lambda obj: obj.dataframe.drop(columns=['Model Path', 'train timestamp'], inplace=True), all_factories))
    except:
        print(['Model Path', 'train timestamp'], 'not found in df')
    if len(model_folders) > 0:
        if save_to_all_dirs:
            ps = list(set([os.path.split(mf)[0] for mf in model_folders]))  # GEt all basepaths
            print(ps)
        else:
            ps = [root_path]
        """
        If if models from multiple test sessions are run, put a csv into every of their basepath folders.
        """
        label = 'scores'
        label += '-avg' if average_only else ''
        label += '-long' if long_tables else ''
        for p in ps:
            print(f"Saving metrics to: {p}")
            # f1_dff.to_csv(p, f'f1-score-dataloader{data_loader_index}.csv')
            # prec_dff.to_csv(p, f'precision-score-dataloader{data_loader_index}.csv')
            # rec_dff.to_csv(p, f'recall-score-dataloader{data_loader_index}.csv')
            f1_dff.to_latex(p, f'f1-{label}-dataloader{data_loader_index}.tex', caption='F1 Scores',
                            label='tbl:f1' + label, long_tables=long_tables, only_tabular_environment=True)
            prec_dff.to_latex(p, f'precision-{label}-dataloader{data_loader_index}.tex', caption='Precision Scores',
                              label='tbl:precision' + label, long_tables=long_tables, only_tabular_environment=True)
            rec_dff.to_latex(p, f'recall-{label}-dataloader{data_loader_index}.tex', caption='Precision Scores',
                             label='tbl:recall' + label, long_tables=long_tables, only_tabular_environment=True)
            # cst_acc_dff.to_csv(p, f'Custom Accuracy{data_loader_index}.tex')
            classfit_dff.to_latex(p, f'Custom Accuracy (Class Fit){label}-dataloader-{data_loader_index}.tex',
                                  caption='Class Fit Scores', label='tbl:classfit' + label, long_tables=long_tables,
                                  only_tabular_environment=True)
            zerofit_dff.to_latex(p, f'Custom Accuracy (Zero Fit){label}-dataloader-{data_loader_index}.tex',
                                 caption='Zero Fit Scores', label='tbl:zerofit' + label, long_tables=long_tables,
                                 only_tabular_environment=True)
            auc_dff.to_latex(p, f'AUC-{label}-dataloader{data_loader_index}.tex', caption='AUC score',
                             label='tbl:auc' + label, long_tables=long_tables, only_tabular_environment=True)
        # make attribute parallel lines plot at this position

    return model_thresholds


def create_lowlabel_plots(model_folders, filename, title_add='', save_to='/home/julian/Documents/projekt-master/bilder',
                          data_loader_index=0):
    TEST_SET = 0;
    VAL_SET = 1;
    TRAIN_SET = 2

    model_thresholds = calculate_best_thresholds(model_folders, data_loader_index=VAL_SET)

    f1_dff = DataFrameFactory()
    prec_dff = DataFrameFactory()
    rec_dff = DataFrameFactory()
    classfit_dff = DataFrameFactory()
    zerofit_dff = DataFrameFactory()
    auc_dff = DataFrameFactory()

    average_only = True

    for mi, model_folder in enumerate(model_folders):
        try:
            if model_thresholds[mi] is None:
                print(f"Encountered nan value in {model_folder} prediction!. Skipping this model.")
                continue
            model_name = os.path.split(model_folder)[1]
            model_name = '.'.join(model_name.split('.')[
                                  -2:]) if '.' in model_name else model_name  # fullname(store_models.load_model_architecture(extract_model_files_from_dir(model_folder)[0][0]))
            model_name = long_to_shortname(model_name)
            print(model_name)
            binary_labels, classes = m.read_binary_label_csv_from_model_folder(model_folder,
                                                                               data_loader_index=data_loader_index)
            predictions, pred_classes = m.read_output_csv_from_model_folder(model_folder,
                                                                            data_loader_index=data_loader_index)
            binary_predictions = m.convert_pred_to_binary(predictions, model_thresholds[mi])
            auc_dff.append(create_metric_score_dataframe(binary_labels, predictions, classes, m.auc_scores, model_name,
                                                         average_only))

        except FileNotFoundError as e:  # folder with not the correct csv?
            print(e)
    auc_dff.natsort_single_index()

    def find_splits_in_string(s):
        if 'train-test-splits_' in s:
            sfile = s.split('train-test-splits_')[1].split('|')[0]
            return (sfile, s.replace('train-test-splits_' + sfile, '').replace('||', '|').replace('||', '|'))
        elif 'train-test-splits-' in s:
            sfile = s.split('train-test-splits-')[1].split('|')[0]
            return (sfile, s.replace('train-test-splits-' + sfile, '').replace('||', '|').replace('||', '|'))
        else:
            return ("standard", s)

    auc_dff.dataframe.index = pd.MultiIndex.from_tuples(
        [find_splits_in_string(i) for i, _ in auc_dff.dataframe.iterrows()])
    groups = auc_dff.dataframe.groupby(level=[1], as_index=False)
    plotm.plot_lowlabel_availabilty(groups, 'Micro average AUC score with low label availability' + title_add, save_to,
                                    filename + 'micro.png', data_col='micro')
    plotm.plot_lowlabel_availabilty(groups, 'Macro average AUC score with low label availability' + title_add, save_to,
                                    filename + 'macro.png', data_col='macro')


def create_parallel_plots(model_folders, savepath, data_loader_index=0, skip_cpc=False, skip_baseline=False):
    TEST_SET = 0;
    VAL_SET = 1;
    TRAIN_SET = 2

    model_thresholds = calculate_best_thresholds(model_folders, data_loader_index=VAL_SET)

    auc_dff = DataFrameFactory()

    average_only = True

    for mi, model_folder in enumerate(model_folders):
        try:
            if model_thresholds[mi] is None:
                print(f"Encountered nan value in {model_folder} prediction!. Skipping this model.")
                continue
            model_name = os.path.split(model_folder)[1]
            model_name = '.'.join(model_name.split('.')[
                                  -2:]) if '.' in model_name else model_name  # fullname(store_models.load_model_architecture(extract_model_files_from_dir(model_folder)[0][0]))
            model_name = long_to_shortname(model_name)
            print(model_name)
            binary_labels, classes = m.read_binary_label_csv_from_model_folder(model_folder,
                                                                               data_loader_index=data_loader_index)
            predictions, pred_classes = m.read_output_csv_from_model_folder(model_folder,
                                                                            data_loader_index=data_loader_index)
            binary_predictions = m.convert_pred_to_binary(predictions, model_thresholds[mi])
            auc_df = create_metric_score_dataframe(binary_labels, predictions, classes, m.auc_scores, model_name,
                                                   average_only)
            # TODO: get train path for call below for most acurate results: not feasible because of random structure
            # TODO: add final layer number desc
            attribute_df = create_model_attribute_table([model_folder], filename=None, skip_cpc=skip_cpc,
                                                        skip_baseline=skip_baseline)
            if len(attribute_df) == 0:
                continue
            attribute_df.index = auc_df.index
            # seperate_folders = model_folder.split(os.sep)
            # train_folder_ix = seperate_folders.index('models')+2
            # train_folder = os.sep.join(seperate_folders[train_folder_ix:])
            auc_dff.append(pd.concat([attribute_df, auc_df], axis=1))

        except FileNotFoundError as e:  # folder with not the correct csv?
            print(e)
    auc_dff.natsort_single_index()
    auc_dff.dataframe.to_csv('ALLMODELSATTRIBUTES' + str(skip_cpc) + '.csv')
    plotm.plot_parallel_coordinates(auc_dff.dataframe, 'micro', savepath + 'micro.png',
                                    drop_columns=['use_class_weights'], put_last_columns=['micro'])
    plotm.plot_parallel_coordinates(auc_dff.dataframe, 'macro', savepath + 'macro.png',
                                    drop_columns=['use_class_weights'], put_last_columns=['macro'])
    # auc_dff.dataframe.to_csv('/home/julian/Desktop/attributeswithauc.csv')


def create_model_attribute_table(model_folders, filename, save_to='/home/julian/Documents/projekt-master/tables',
                                 skip_cpc=True, skip_baseline=False):
    attribute_df = DataFrameFactory()

    for f in model_folders:
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir (model?)
                name = long_to_shortname(root.split(os.sep)[-1].split('|')[0])
                attrs = extract_model_attributes(root, name, skip_cpc, skip_baseline)
                if attrs == None: #got skipped
                    continue
                attrs['Model Name'] = name
                attrs = pd.DataFrame(attrs, index=[0])
                attrs = attrs.set_index('Model Name')
                attribute_df.append(attrs)
    attribute_df.dataframe.drop_duplicates(inplace=True)
    attribute_df.natsort_single_index()
    if not filename is None:
        attribute_df.dataframe.to_csv('/home/julian/Desktop/' + filename + '.csv')
    return attribute_df.dataframe

def extract_model_attributes(model_folder, model_name, skip_cpc=True, skip_baseline=True, short_key=False):
    is_cpc = True
    attrs = {}
    try:
        #model_arch.txt
        with open(os.path.join(model_folder, 'model_arch.txt'), 'r') as file:
            content = file.read()
        if 'BaselineNet' in content:
            is_cpc = False
        attrs["Model Path"] = model_folder
        if (skip_cpc and is_cpc) or (skip_baseline and not is_cpc):
            return None
        with open(os.path.join(model_folder, 'train_params.txt'), 'r') as file:
                content = file.read()
                attrs['Uses Classweights'] = not 'use_class_weights=False' in content
                if 'norm_fn=' in content:
                    attrs["Train Normalization Function"] = content.split('norm_fn=')[1].split(',')[0]
                else:
                    attrs["Train Normalization Function"] = 'normalize_std_scaling'
                if 'splits_file=' in content:
                    attrs['Train splitsfile'] = content.split("splits_file='")[1].split("'")[0].replace('.txt', '')
                else:
                    attrs['Train splitsfile'] = 'train-test-splits'
                if 'crop_size=' in content:
                    attrs["Crop Size"] = content.split('crop_size=')[1].split(',')[0]

        #params.txt
        fname = pathlib.Path(os.path.join(model_folder, 'params.txt'))
        timestamp = fname.stat().st_mtime
        attrs['train timestamp'] = datetime.fromtimestamp(timestamp)
        with open(os.path.join(model_folder, 'params.txt'), 'r') as file:
            content = file.read()
            if 'norm_fn=' in content:
                attrs["Test Normalization Function"] = content.split('norm_fn=')[1].split(',')[0]
            else:
                if int(timestamp) > 1637682016: #oh god
                    attrs["Test Normalization Function"] = 'normalize_std_scaling'
                else:
                    attrs["Test Normalization Function"] = 'normalize_minmax_scaling_different'

            # if 'splits_file=' in content:
            #     attrs['Test splitsfile'] = content.split("splits_file='")[1].split("'")[0].replace('.txt', '')
            # else:
            #     attrs['Test splitsfile'] = 'train-test-splits'

        if is_cpc:
            with open(os.path.join(model_folder, 'model_variables.txt'), 'r') as file:
                content = file.readlines()
            for i, line in enumerate(content):
                if '{' in line:
                    content = '\n'.join(content[i:])
                    break
            data = json.loads(content)
            attrs['strided'] = 'cpc_encoder_as_strided.StridedEncoder"' in content
            attrs['Freeze CPC'] = not '"freeze_cpc": false' in content
            attrs['uses Context'] = '"use_context": true' in content
            attrs['uses Latents'] = '"use_latents": true' in content
            attrs['normalizes latents'] = '"normalize_latents": true' in content
            if "sampling_mode" in content:
                attrs['CPC Sampling Mode'] = content.split('"sampling_mode": ')[1].split(',')[0][1:-1]
            else:
                attrs['CPC Sampling Mode'] = 'same'
            try:
                cpc_type = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"].keys())[0]
                attrs['CPC Type'] = cpc_type.split('.')[-2]
                attrs['Autoregressive'] = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"][cpc_type]["_modules"][""]["autoregressive"])[0].split('.')[-2]
                attrs['Encoder'] = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"][cpc_type]["_modules"][""]["encoder"])[0].split('.')[-2]
                attrs['Predictor'] = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"][cpc_type]["_modules"][""]["predictor"])[0].split('.')[-2]
            except KeyError:
                print("Model does not follow CPC Combined architecture spec.")
            if '"downstream_model":' in content:
                attrs['Downstream Model'] = \
                content.split('"downstream_model": {')[1].split('": {')[0].strip().lstrip('"').split('.')[
                    -2]

            # params.txt
            with open(os.path.join(model_folder, 'train_params.txt'), 'r') as file:
                content = file.read()
            if 'pretrain_epochs' in content and is_cpc:
                epos = content.split('pretrain_epochs=')[1].split(',')[0]
                attrs['Pretrain Epochs'] = epos
            if 'downstream_epochs' in content:
                epos = content.split('downstream_epochs=')[1].split(',')[0]
                attrs['Downstream Epochs'] = epos
            attrs['Latent Size'] = content.split('latent_size=')[1].split(',')[0]
            attrs['Context Size'] = content.split('hidden_size=')[1].split(',')[0]
            attrs['timesteps in'] = content.split('timesteps_in=')[1].split(',')[0]
            attrs['timesteps out'] = content.split('timesteps_out=')[1].split(',')[0]


        else:  # not cpc
            with open(os.path.join(model_folder, 'train_params.txt'), 'r') as file:
                content = file.read()
                if 'downstream_epochs' in content:
                    epos = content.split('downstream_epochs=')[1].split(',')[0]
                    attrs['Downstream Epochs'] = epos

            with open(os.path.join(model_folder, 'model_variables.txt'), 'r') as file:
                content = file.readlines()
                for i, line in enumerate(content):
                    if '{' in line:
                        content = '\n'.join(content[i:])
                        break
            data = json.loads(content)
            attrs['Convolutional Layer Number'] = str(count_key_in_dict(data, 'torch.nn.modules.conv.Conv1d'))
            attrs['uses Max Pool'] = 'torch.nn.modules.pooling.MaxPool1d' in content
            attrs['uses Adaptive Average Pooling'] = 'torch.nn.modules.pooling.AdaptiveAvgPool1d' in content
            attrs['uses Linear'] = 'torch.nn.modules.linear.Linear' in content
            attrs['uses LSTM'] = 'torch.nn.modules.rnn.LSTM' in content
            attrs['uses BatchNorm'] = 'torch.nn.modules.batchnorm.BatchNorm1d' in content
            attrs['Sum of Strides'] = str(int(np.array(extract_values_for_key_in_dict(data, 'stride')).sum()))
            attrs['Sum of Dilation'] = str(int(np.array(extract_values_for_key_in_dict(data, 'dilation')).sum()))
            attrs['Sum of Paddings'] = str(int(np.array(extract_values_for_key_in_dict(data, 'padding')).sum()))
            attrs['Sum of Filters'] = str(int(np.array(extract_values_for_key_in_dict(data, 'kernel_size')).sum()))
            final_layers = {'baseline_FCN': '3',
                            'baseline_cnn_v0_2': '4',
                            'baseline_cnn_v2': '3',
                            'baseline_cnn_v6': '4',
                            'baseline_cnn_v1': '3',
                            'baseline_cnn_v5': '4',
                            'baseline_cnn_v4': '4',
                            'baseline_cnn_v14': '4',
                            'baseline_cnn_v3': '4',
                            'baseline_cnn_v0_1': '4',
                            'baseline_cnn_v0': '4',
                            'baseline_cnn_v9': '4',
                            'baseline_cnn_v8': '4',
                            'baseline_cnn_v0_3': '4',
                            'baseline_TCN_down': '4',
                            'baseline_TCN_flatten': '1',
                            'baseline_cnn_v7': '4',
                            'baseline_TCN_block': '1',
                            'baseline_rnn_simplest_lstm': '5',
                            'baseline_rnn_simplest_gru': '5',
                            'baseline_MLP': '2',
                            'baseline_alex_v2': '1',
                            'baseline_cnn_v15': '4',
                            'baseline_TCN_last': '2'}
            try:
                attrs['Final Layer'] = final_layers[list(data.keys())[0].split('.')[-2]]
            except KeyError:
                final_layers_short = {'BL_FCN': '3',
                                'BL_v0_2': '4',
                                'BL_v2': '3',
                                'BL_v6': '4',
                                'BL_v1': '3',
                                'BL_v5': '4',
                                'BL_v4': '4',
                                'BL_v14': '4',
                                'BL_v3': '4',
                                'BL_v0_1': '4',
                                'BL_v0': '4',
                                'BL_v9': '4',
                                'BL_v8': '4',
                                'BL_v0_3': '4',
                                'BL_TCN_down': '4',
                                'BL_TCN_flatten': '1',
                                'BL_v7': '4',
                                'BL_TCN_block': '1',
                                'BL_rnn_simplest_lstm': '5',
                                'BL_MLP': '2',
                                'BL_alex_v2': '1',
                                'BL_v15': '4',
                                'BL_TCN_last': '2'}
                try:
                    attrs["Final Layer"] = final_layers_short[model_name]
                except KeyError:
                    attrs["Final Layer"] = None
                    print(model_name, 'not found in final layers dict')
    except FileNotFoundError:
        print(model_folder, 'is not a model folder?')
    return attrs


def sort_naturally(data):
    pass


if __name__ == '__main__':
    TEST_SET = 0;
    VAL_SET = 1;
    TRAIN_SET = 2
    paths = [
        # '/home/julian/Downloads/Github/contrastive-predictive-coding/models/16_12_21-15-09-test|(10x)bl_TCN_down+(10x)bl_cnn_v1+(10x)bl_cnn_v14+(10x)bl_cnn_v15+(10x)bl_cnn_v8'
        #'/home/julian/Downloads/Github/contrastive-predictive-coding/models/16_08_21-10-16-test|(40x)cpc/14_08_21-15_36-train|(4x)cpc/architectures_cpc.cpc_combined.CPCCombined2|train-test-splits-fewer-labels60|use_weights|unfrozen|C|m:all|cpc_downstream_cnn'
        #'/home/julian/Downloads/Github/contrastive-predictive-coding/models/17_12_21-13-13-test|(160x)cpc',
        #'/home/julian/Downloads/Github/contrastive-predictive-coding/models/16_12_21-15-09-test|(10x)bl_TCN_down+(10x)bl_cnn_v1+(10x)bl_cnn_v14+(10x)bl_cnn_v15+(10x)bl_cnn_v8',
        # '/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_12_21-13-40-test|(32x)cpc',
        # ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_12_21-16-23-test|(4x)bl_cnn_v14+(4x)bl_cnn_v8',
        # '/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_01_22-13-43-test|(12x)cpc']
        # '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_22-16-33-test|(6x)cpc'
        # ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/18_01_22-15-28-test|(28x)cpc/14_01_22-13-26-train|(14x)cpc',
        # "/home/julian/Downloads/Github/contrastive-predictive-coding/models/18_01_22-15-28-test|(28x)cpc/14_01_22-15-39-train|(14x)cpc"],
        '/home/julian/Downloads/Github/contrastive-predictive-coding/models/',
    ]
    root_path = '/home/julian/Desktop/'
    # paths = ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_12_21-16-37-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/27_12_21-14-01-test|(8x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_12_21-20-09-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/05_01_22-17-30-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_01_22-16-33-test|(6x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/30_11_21-16-28-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/15_11_21-13-25-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/06_12_21-19-40-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_12_21-14-17-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-15-12-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/06_01_22-15-01-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_12_21-15-44-test|(6x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/24_12_21-11-14-test|(8x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/01_12_21-18-14-test|(12x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/10_01_22-17-11-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/30_11_21-18-25-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-13-42-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/01_12_21-19-56-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/04_01_22-18-21-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_11_21-13-23-test|(8x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_11_21-21-41-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/11_11_21-17-24-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/16_12_21-13-17-test|(12x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/06_01_22-18-55-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/06_01_22-18-04-test|(3x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/01_12_21-12-48-test|(12x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/12_11_21-16-02-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-11-09-test|(2x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-16-48-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-13-13-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_11_21-21-38-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_12_21-13-40-test|(32x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_11_21-19-25-test|(16x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/23_12_21-15-00-test|(3x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/02_12_21-17-56-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-12-26-test|(4x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-14-27-test|cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/13_12_21-16-04-test|(3x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/17_12_21-13-13-test|(160x)cpc', '/home/julian/Downloads/Github/contrastive-predictive-coding/models/07_01_22-17-01-test|(4x)cpc']
    for pl in paths:
        if type(pl) == list:
            model_folders = [a for p in pl for a in auto_find_tested_models_recursive(p)]
        else:
            model_folders = auto_find_tested_models_recursive(pl)
        # # low_label_noclassweights_paths = ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_07_21-17-50-test|(48x)cpc',
        # #                                '/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_07_21-16-test|(5x)bl_TCN_down+(5x)bl_cnn_v1+(5x)bl_cnn_v14+(5x)bl_cnn_v15+(5x)bl_cnn_v8']
        # # model_folders = [a for p in low_label_noclassweights_paths for a in auto_find_tested_models_recursive(p)]
        # # create_lowlabel_plots(model_folders, data_loader_index=TEST_SET, filename='low_label_availability_noclassweights')
        # # low_label_classweights_paths_more_epochs = ['models/11_08_21-15-58-test|(10x)bl_TCN_down+(10x)bl_cnn_v1+(10x)bl_cnn_v14+(10x)bl_cnn_v15+(10x)bl_cnn_v8',
        # #                                 #'models/13_08_21-10-47-test|(80x)cpc',
        # #                                 '/home/julian/Downloads/Github/contrastive-predictive-coding/models/16_08_21-10-16-test|(40x)cpc']
        # # model_folders = [a for p in low_label_classweights_paths_more_epochs for a in auto_find_tested_models_recursive(p)]
        # # create_lowlabel_plots(model_folders, data_loader_index=TEST_SET, filename='low_label_availability_classweights-more-epochs', title_add=' (50 CPC epochs, )')
        # #
        # # low_label_classweights_paths = ['models/11_08_21-15-58-test|(10x)bl_TCN_down+(10x)bl_cnn_v1+(10x)bl_cnn_v14+(10x)bl_cnn_v15+(10x)bl_cnn_v8',
        # #                                 'models/13_08_21-10-47-test|(80x)cpc']
        # # model_folders = [a for p in low_label_classweights_paths for a in auto_find_tested_models_recursive(p)]
        # #create_lowlabel_plots(model_folders, data_loader_index=TEST_SET, filename='low_label_availability_classweights', title_add=' (20 CPC epochs)', save_to=paths[0])
        #
        # # low_label_classweights_paths = ['/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_07_21-18-41-test|(48x)cpc',
        # #                                 '/home/julian/Downloads/Github/contrastive-predictive-coding/models/20_07_21-17-test|(5x)bl_TCN_down+(5x)bl_cnn_v1+(5x)bl_cnn_v14+(5x)bl_cnn_v15+(5x)bl_cnn_v8']
        # # model_folders = [a for p in low_label_classweights_paths for a in auto_find_tested_models_recursive(p)]
        # # create_lowlabel_plots(model_folders, data_loader_index=TEST_SET, filename='low_label_availability_classweights')
        #
        # # model_folders = auto_find_tested_models_recursive('/home/julian/Downloads/Github/contrastive-predictive-coding/models/')
        #
        create_paper_metrics(model_folders, root_path=root_path, data_loader_index=TEST_SET, average_only=True, cut_name_attrs=True, use_attrs_in_name=False, include_attrs_in_table=True, save_to_all_dirs=False) #Old
        #create_paper_metrics(model_folders, root_path='', data_loader_index=TEST_SET, average_only=True, use_attrs_in_name=True, save_to_all_dirs=True, include_attrs_in_table=True)  # On Testset
        #create_paper_plots(model_folders, data_loader_index=TEST_SET)

    # create_paper_metrics(model_folders, root_path=path, data_loader_index=TEST_SET, average_only=True, save_to_all_dirs=False) #On Testset
    # create_paper_metrics(model_folders, root_path=path, data_loader_index=TEST_SET, average_only=True, long_tables=True, save_to_all_dirs=False)
    # create_model_attribute_table(model_folders, 'baseline-attributes-full', skip_cpc=True, skip_baseline=False)
    # create_model_attribute_table(model_folders, 'cpc-attributes', skip_cpc=False, skip_baseline=True)
    # create_parallel_plots(model_folders, '/home/julian/Desktop/cpc-attributes-parallelcoords', skip_cpc=False, skip_baseline=True)
    # create_parallel_plots(model_folders, '/home/julian/Desktop/bl-attributes-parallelcoords', skip_cpc=True, skip_baseline=False)
