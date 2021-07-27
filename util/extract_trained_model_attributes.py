import itertools
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from pandas.plotting import parallel_coordinates
import plotly.express as px
from scipy.interpolate import make_interp_spline

from util.utility.dict_utils import count_key_in_dict, extract_values_for_key_in_dict


def extract_baseline_model_attributes(train_folders, extract_baselines=True, extract_cpcs=False, exclude_models=[]):
    model_descriptions = {}
    collision_counter = 1
    for f in train_folders:
        if not ('test' in f):
            print(f"Checking {f}...")
            for root, dirs, files in os.walk(f, followlinks=True):
                print(root)
                if len(dirs) == 0 and len(files)==0:
                    continue
                if len(dirs) > 0: #Traverse more
                    continue
                if len(dirs) == 0 and len(files)>0: #leaf dir (model?)
                    name = os.path.basename(root).split('|')[0]
                    print(root)
                    if 'architectures_cpc.' in name and not extract_cpcs: #its a cpc model but dont extract
                        continue
                    if 'architectures_baseline_challenge.' in name and not extract_baselines: #its a cpc model but dont extract
                        continue
                    if any([em in name for em in exclude_models]): #exclude this model
                        continue
                    name = name.replace('architectures_baseline_challenge.', '')
                    name = name.replace('architectures_cpc.', '')
                    print(name)
                    if name in model_descriptions:
                        name = name + f'+{collision_counter}'
                        collision_counter += 1
                    model_descriptions[name]={}
                    if 'model_arch.txt' in files:

                        with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                            content = file.read()
                        #print(content)

                    if 'model_variables.txt' in files:
                        with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                            content = file.readlines()
                            for i, line in enumerate(content):
                                if '{' in line:
                                    content = '\n'.join(content[i:])
                                    break
                        data = json.loads(content)
                        model_descriptions[name]['Conv1d'] = count_key_in_dict(data, 'torch.nn.modules.conv.Conv1d')
                        model_descriptions[name]['MaxPool1d'] = count_key_in_dict(data, 'torch.nn.modules.pooling.MaxPool1d')
                        model_descriptions[name]['AdaptiveAvgPool1d'] = count_key_in_dict(data, 'torch.nn.modules.pooling.AdaptiveAvgPool1d')
                        model_descriptions[name]['Linear'] = count_key_in_dict(data, 'torch.nn.modules.linear.Linear')
                        model_descriptions[name]['LSTM'] = count_key_in_dict(data, 'torch.nn.modules.rnn.LSTM')
                        model_descriptions[name]['BatchNorm1d'] = count_key_in_dict(data, 'torch.nn.modules.batchnorm.BatchNorm1d')
                        model_descriptions[name]['stride product'] = np.array(extract_values_for_key_in_dict(data, 'stride')).prod()
                        model_descriptions[name]['dilation sum'] = np.array(extract_values_for_key_in_dict(data, 'dilation')).sum()
                        model_descriptions[name]['padding sum'] = np.array(extract_values_for_key_in_dict(data, 'padding')).sum()
                        model_descriptions[name]['kernelsize sum'] = np.array(extract_values_for_key_in_dict(data, 'kernel_size')).sum()
                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()
                        model_descriptions[name]['uses class weights'] = 'use_class_weights=True' in content

                    if model_descriptions[name] == {}:
                        del model_descriptions[name]
    return model_descriptions

def extract_cpc_model_attributes(train_folders, exclude_models=[]):
    model_descriptions = {}
    for f in train_folders:
        print(f"Checking {f}...")
        for root, dirs, files in os.walk(f, followlinks=True):
            print(root)
            if len(dirs) == 0 and len(files)==0:
                continue
            if len(dirs) > 0: #Traverse more
                continue
            if len(dirs) == 0 and len(files)>0: #leaf dir (model?)
                name = os.path.basename(root).split('|')[0]
                print(root)
                if 'architectures_baseline_challenge.' in name: #its a baseline model so dont extract
                    continue
                if any([em in name for em in exclude_models]): #exclude this model
                    continue
                name = name.replace('architectures_baseline_challenge.', '')
                name = name.replace('architectures_cpc.', '')
                print(name)
                model_descriptions[name]={}
                if 'model_arch.txt' in files:

                    with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                        content = file.read()
                    #print(content)

                if 'model_variables.txt' in files:
                    with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                        content = file.readlines()
                        for i, line in enumerate(content):
                            if '{' in line:
                                content = '\n'.join(content[i:])
                                break
                    data = json.loads(content)
                    model_descriptions[name]['weights frozen'] = not '"freeze_cpc": false' in content
                    model_descriptions[name]['use context'] = '"use_context": true' in content
                    model_descriptions[name]['use latents'] = '"use_latents": true' in content
                    model_descriptions[name]['normalized latents'] = '"normalize_latents": true' in content

                    if "sampling_mode" in content:
                        model_descriptions[name]['sampling mode'] = content.split('"sampling_mode": ')[1].split(',')[0][1:-1]
                    else:
                        model_descriptions[name]['sampling mode'] = 'none'
                    if '"downstream_model":' in content:
                        model_descriptions[name]['downstream model'] = content.split('"downstream_model": {')[1].split('": {')[0].strip().lstrip('"').split('.')[-2]
                    else:
                        model_descriptions[name]['downstream model'] = 'none'

                if 'params.txt' in files:
                    with open(os.path.join(root, 'params.txt'), 'r') as file:
                        content = file.read()
                    model_descriptions[name]['uses class weights'] = 'use_class_weights=True' in content
                if model_descriptions[name] == {}:
                    del model_descriptions[name]
    return model_descriptions



def plot_parallel_coordinates(df, name_column):
    # import packages
    import numpy as np
    import matplotlib.pyplot as plt
    plt.tight_layout()
    model_names = df[name_column].values
    df = df.drop(name_column, axis='columns')
    data = df.values
    n_categories = data.shape[1]
    x = np.arange(n_categories)
    x_labels = df.columns

    fig, axs = plt.subplots(1, n_categories-1, sharey='none')

    # plot subplots and set xlimit
    for i in range(len(axs)):
        for j in range(len(data)):
            axs[i].plot(x_labels, data[j])
        axs[i].set_xlim(x_labels[i], x_labels[i+1])
        axs[i].set_ylim(data[:, i].min(), data[:, i].max())
        axs[i].tick_params('x', labelrotation=90)
    # set width space to zero
    plt.subplots_adjust(wspace=0)
    # show plot
    plt.show()


def parallel_coordinates_custom(df, name_column, style=None): #https://stackoverflow.com/q/8230638

    model_names = df[name_column].values
    df = df.drop(name_column, axis='columns')
    data_sets = df.values
    n_categories = data_sets.shape[1]
    x = np.arange(n_categories)
    x_labels = list(df.columns)


    dims = len(data_sets[0])
    x    = range(dims)#x_labels#
    fig, axes = plt.subplots(1, dims-1, sharey='none')

    if style is None:
        style = ['r-']*len(data_sets)

    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        if type(m) == str:
            sorted(m)
        else:
            mn = min(m)
            mx = max(m)
            if mn == mx:
                mn -= 0.5
                mx = mn + 1.
            r  = float(mx - mn)
            min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) /
                min_max_range[dimension][2]
                for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi])
        ax.set_xlim([x[i], x[i+1]])

    # Set the x axis ticks
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in range(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)


    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in range(ticks)]
    axx.set_yticklabels(labels)

    # Stack the subplots
    plt.subplots_adjust(wspace=0)

    plt.show()

def factorize_dataframe(df, column):
    print(df[column])
    codes = df[column].astype('category').cat.codes
    print('after fac', df[column])
    df.update(pd.DataFrame({column: codes}))

#https://github.com/jraine/parallel-coordinates-plot-dataframe/blob/master/parallel_plot.py
def parallel_plot(df,cols,rank_attr,cmap='Spectral',spread=None,curved=False,curvedextend=0.1):
    '''Produce a parallel coordinates plot from pandas dataframe with line colour with respect to a column.
    Required Arguments:
        df: dataframe
        cols: columns to use for axes
        rank_attr: attribute to use for ranking
    Options:
        cmap: Colour palette to use for ranking of lines
        spread: Spread to use to separate lines at categorical values
        curved: Spline interpolation along lines
        curvedextend: Fraction extension in y axis, adjust to contain curvature
    Returns:
        x coordinates for axes, y coordinates of all lines'''
    colmap = matplotlib.cm.get_cmap(cmap)
    cols = cols + [rank_attr]

    fig, axes = plt.subplots(1, len(cols)-1, sharey=False, figsize=(3*len(cols)+3,5))
    valmat = np.ndarray(shape=(len(cols),len(df)))
    x = np.arange(0,len(cols),1)
    ax_info = {}
    for i,col in enumerate(cols):
        vals = df[col]
        if (vals.dtype == float) & (len(np.unique(vals)) > 20):
            minval = np.min(vals)
            maxval = np.max(vals)
            rangeval = maxval - minval
            vals = np.true_divide(vals - minval, maxval-minval)
            nticks = 5
            tick_labels = [round(minval + i*(rangeval/nticks),4) for i in range(nticks+1)]
            ticks = [0 + i*(1.0/nticks) for i in range(nticks+1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels,ticks]
        else:
            vals = vals.astype('category')
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = 0
            maxval = len(cats)-1
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval-minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels,ticks]
            if spread is not None:
                offset = np.arange(-1,1,2./(len(c_vals)))*2e-2
                np.random.shuffle(offset)
                c_vals = c_vals + offset
            valmat[i] = c_vals

    extendfrac = curvedextend if curved else 0.05
    for i,ax in enumerate(axes):
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x)*20)
                a_BSpline = make_interp_spline(x, valmat[:,idx],k=3,bc_type='clamped')
                y_new = a_BSpline(x_new)
                ax.plot(x_new,y_new,color=colmap(valmat[-1,idx]),alpha=0.3)
            else:
                ax.plot(x,valmat[:,idx],color=colmap(valmat[-1,idx]),alpha=0.3)
        ax.set_ylim(0-extendfrac,1+extendfrac)
        ax.set_xlim(i,i+1)

    for dim, (ax,col) in enumerate(zip(axes,cols)):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])
        ax.set_xticklabels([cols[dim]])


    plt.subplots_adjust(wspace=0)
    norm = matplotlib.colors.Normalize(0,1)#*axes[-1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm,pad=0,ticks=ax_info[rank_attr][1],extend='both',extendrect=True,extendfrac=extendfrac)
    if curved:
        cbar.ax.set_ylim(0-curvedextend,1+curvedextend)
    cbar.ax.set_yticklabels(ax_info[rank_attr][0])
    cbar.ax.set_xlabel(rank_attr)
    plt.show()

    return x,valmat


if __name__ == '__main__':
    model_descriptions = extract_baseline_model_attributes(['/home/julian/Downloads/Github/contrastive-predictive-coding/models_symbolic_links/train'], exclude_models=['v14'])
    #model_descriptions = extract_cpc_model_attributes(['/home/julian/Downloads/Github/contrastive-predictive-coding/models_symbolic_links/train/correct-age/no_class_weights/cpc'])
    df = pd.DataFrame.from_dict(model_descriptions).transpose()
    df.index.name="Model"
    cols = list(df.columns)
    df['micro auc'] = np.random.randn(len(df))
    parallel_plot(df.reset_index(), list(set(df.columns)-{'micro auc'}), 'micro auc', curved=True)
    plt.show()
    #plot_parallel_coordinates(df.reset_index(), 'Model')
    # parallel_coordinates_custom(df.reset_index(), 'Model')
    # df.to_csv('/home/julian/Desktop/descriptions.csv')
    # fig = px.parallel_coordinates(df.reset_index(), color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
    # fig.write_image('tmp.png')

