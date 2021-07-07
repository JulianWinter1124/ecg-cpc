import itertools
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import plotly.express as px

def extract_baseline_model_attributes(train_folders):
    model_descriptions = {}
    for f in train_folders:
        if not ('test' in f):
            print(f"Checking {f}...")
            for root, dirs, files in os.walk(f):
                print(root)
                if len(dirs) == 0 and len(files)==0:
                    continue
                if len(dirs) > 0: #Traverse more
                    continue
                if len(dirs) == 0 and len(files)>0: #leaf dir (model?)
                    print(root)
                    name = os.path.basename(root).split('|')[0].replace('architectures_baseline_challenge.', '')
                    print(name)
                    model_descriptions[name]={}
                    if 'model_arch.txt' in files:

                        with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                            content = file.read()
                        #print(content)

                    if 'model_variables.txt' in files:
                        with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                            content = file.readlines()
                            content = '\n'.join(content[2:])
                        data = json.loads(content)
                        model_descriptions[name]['Conv1d'] = count_key_in_dict(data, 'torch.nn.modules.conv.Conv1d')
                        model_descriptions[name]['MaxPool1d'] = count_key_in_dict(data, 'torch.nn.modules.pooling.MaxPool1d')
                        model_descriptions[name]['AdaptiveAvgPool1d'] = count_key_in_dict(data, 'torch.nn.modules.pooling.AdaptiveAvgPool1d')
                        model_descriptions[name]['Linear'] = count_key_in_dict(data, 'torch.nn.modules.linear.Linear')
                        model_descriptions[name]['LSTM'] = count_key_in_dict(data, 'torch.nn.modules.rnn.LSTM')
                        model_descriptions[name]['GRU'] = count_key_in_dict(data, 'torch.nn.modules.rnn.GRU')
                        model_descriptions[name]['BatchNorm1d'] = count_key_in_dict(data, 'torch.nn.modules.batchnorm.BatchNorm1d')
                        model_descriptions[name]['stride product'] = np.array(extract_values_for_key_in_dict(data, 'stride')).prod()
                        model_descriptions[name]['dilation sum'] = np.array(extract_values_for_key_in_dict(data, 'dilation')).sum()
                        model_descriptions[name]['padding sum'] = np.array(extract_values_for_key_in_dict(data, 'padding')).sum()
                        model_descriptions[name]['kernelsize sum'] = np.array(extract_values_for_key_in_dict(data, 'kernel_size')).sum()
                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()

                    if model_descriptions[name] == {}:
                        del model_descriptions[name]
    return model_descriptions



def count_key_in_dict(dictionary, search_key, decision_fn=None):
    count = 0
    if type(dictionary) == dict:
        for k, v in dictionary.items():
            if decision_fn is None:
                count += k == search_key
            else:
                count += decision_fn(k, search_key)
            count += count_key_in_dict(v, search_key)
    return count

def extract_values_for_key_in_dict(dictionary, search_key, decision_fn=None):
    values = []
    if type(dictionary) == dict:
        for k, v in dictionary.items():
            if decision_fn is None:
                if k == search_key:
                    if type(v) == list:
                        values += v
                    else:
                        values += [v]
            elif decision_fn(k, search_key):
                if type(v) == list:
                    values += v
                else:
                    values += [v]
            values += extract_values_for_key_in_dict(v, search_key)
    return values



if __name__ == '__main__':
    model_descriptions = extract_baseline_model_attributes(['/home/julian/Downloads/Github/contrastive-predictive-coding/models/25_06_21-12-train|+bl_MLP+bl_TCN_block+bl_TCN_down+bl_TCN_flatten+bl_TCN_last+bl_alex_v2+bl_cnn_v15+bl_cnn_v7+bl_rnn_simplest_lstm',
                                       '/home/julian/Downloads/Github/contrastive-predictive-coding/models/21_05_21-11-train|bl_cnn_v0+bl_cnn_v0_1+bl_cnn_v0_2+bl_cnn_v0_3+bl_cnn_v1+bl_cnn_v14+bl_cnn_v2+bl_cnn_v3+bl_cnn_v4+bl_cnn_v5+bl_cnn_v6+bl_cnn_v8+bl_cnn_v9'])
    df = pd.DataFrame.from_dict(model_descriptions).transpose()
    df.index.name="Model"
    #df.to_csv('/home/julian/Desktop/descirptions.csv')
    fig = px.parallel_coordinates(df.reset_index(), color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
    fig.write_image('tmp.png')

