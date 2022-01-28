import json
import os
import pathlib
import shutil
from glob import glob
from itertools import chain
from pathlib import Path
import numpy as np
from collections import Counter

from importlib_resources import files

from util.utility.dict_utils import count_key_in_dict, extract_values_for_key_in_dict


def move_incomplete_training_folders(base='.'):
    print(f"Cleaning up.")
    move_to_base = os.path.join(base, 'delete')
    move_to_old_base = os.path.join(base, 'old')
    folders = glob(os.path.join(base, "*", ""))
    for f in folders:
        if 'explain' in f:
            continue
        if Path(f) == Path(move_to_base) or Path(f) == Path(move_to_old_base):
            continue
        for root, dirs, files in os.walk(f):

            move_to = os.path.join(move_to_base, *root.split(os.sep)[1:])
            move_to_old = os.path.join(move_to_old_base, *root.split(os.sep)[1:])
            if (len(dirs) == 0 and len(files) == 0):
                print(f"Found empty dir {root}")
                os.rmdir(root)
            elif (len(dirs) == 0 and len(files) <= 3):
                print(f"Found unsuccessfull run dir {root}")
                move_filetree(root, move_to)
            elif len(dirs) == 0:
                if not any(['_checkpoint_epoch_' in file for file in files]) and any(['_full_model.pt' in file for file in files]): #Check if its training (full model) but no checkpoint
                    print("Found incomplete:", f, "Moving to delete")
                    #print(root, move_to)
                    move_filetree(root, move_to)
                elif 'params.txt' in files:
                    param_file = os.path.join(root, 'params.txt')
                    # print(f"Looking for params.txt at {param_file}")
                    with open(param_file, 'r') as file:
                        content = file.read()
                        if not 'dry_run' in content:
                            print(f"Old model found (no dry_run param)")
                            move_filetree(root, move_to_old)
                        elif 'dry_run=True' in content:
                            # remove
                            print(f'Found dry_run=True. Removing {root}, {dirs}, {files}')
                            move_filetree(root, move_to)
                else:
                    print(f"Could not find params.txt in {root}")
                    move_filetree(root, move_to_old)


def move_filetree(source, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    print(f'Moving contents of {source} to {dest}')
    # remove
    try:
        shutil.move(source, dest, copy_function=shutil.copytree)
    except shutil.Error as e:
        print(f'Moving contents of {source} to {dest} failed. Error{e}')


def rename_folders_into_splits(base='.'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        checks = []
        if not ('-test' in f or '-train' in f or 'splits' in f):
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files) == 0:
                    continue
                if len(dirs) > 0:  # Traverse more
                    continue
                if len(dirs) == 0 and len(files) > 0:  # leaf dir
                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()
                        checks.append('redo_splits' in content)  # Not a test dir

        if len(checks) > 0 and all(checks):  # Found a test dir
            print(f"{f} is a is a model with specific file splits. Renaming")
            new_folder_name = f.rstrip('/') + '|[splits]'
            print(f"Renaming {f} to {new_folder_name}")
            os.rename(f, new_folder_name)


def rename_folders_into_test(base='.'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        checks = []
        if not ('-test' in f or '-train' in f):
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files) == 0:
                    continue
                if len(dirs) > 0:  # Traverse more
                    continue
                if len(dirs) == 0 and len(files) > 0:  # leaf dir
                    checks.append(is_test_folder(files) and not is_train_folder(files))  # Not a test dir

        if len(checks) > 0 and all(checks):  # Found a test dir
            print(f"{f} is a test directory. Renaming")
            os.rename(f, f.rstrip('/') + '-test(a)')


def rename_folders_into_models(base='.', folders=None):
    folders = folders or glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        model_names = []
        for root, dirs, files in os.walk(f):
            if len(dirs) > 0:  # Do not look in leafs
                model_folders = list(filter(lambda x: '.' in x, dirs))  # model folders?
                print(f"Found the following model files: {model_folders} in {root}")
                model_names += map(long_to_abbreviation, model_folders)
        if len(model_names) > 0:
            counts = Counter(model_names)
            model_names_unique = sorted(list(map(lambda x: x[0] if x[1] <= 1 else f"({x[1]}x){x[0]}", counts.items())))
            old_folder_name = f.split('|')[0].rstrip('/')
            new_folder_name = old_folder_name + '|' + '+'.join(model_names_unique)
            print(f"Renaming {f} to {new_folder_name}")
            try:
                os.rename(f, new_folder_name)
            except OSError as ose:
                if ose.errno == 36:
                    os.rename(f, new_folder_name[0:254])
                elif ose.errno == 39:
                    print("Folder already exists... removing")
                    b = os.path.abspath(base)
                    move_filetree(f, os.path.join(b, 'delete', os.path.relpath(f, b)))
                    #
                else:
                    raise


def filter_folders_params(base='.', params_filter='use_class_weights'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir
                if 'params.txt' in files:
                    with open(os.path.join(root, 'params.txt'), 'r') as file:
                        content = file.read()
                    checks.append(params_filter in content)
        if len(checks) > 0 and any(checks):  # Found a test dir
            print(f"{f} Train session used {params_filter} param.")
            filtered += [f]
    return filtered


def filter_folders_model_variables(base='.', model_variables_filter='normalize_latents'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir
                if 'model_variables.txt' in files:
                    with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                        content = file.read()
                    checks.append(model_variables_filter in content)
        if len(checks) > 0 and any(checks):  # Found a test dir
            print(f"{f} Train session used {model_variables_filter} param.")
            filtered += [f]
    return filtered

def filter_folders_universal(base='.', folders=None, file_content_filter_fnlist_dict={}, file_filter_fnlist=[], check_all=True):
    if folders is None:
        folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        content_checks = {}
        for k,v in file_content_filter_fnlist_dict.items():
            content_checks[k]=[False]*len(v)
        file_checks = [False]*len(file_filter_fnlist)
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir
                for checkfile, checkfns in file_content_filter_fnlist_dict.items():
                    if checkfile in files:
                        with open(os.path.join(root, checkfile), 'r') as file:
                            content = file.read()
                        for i, checkfn in enumerate(checkfns):
                            if callable(checkfn):
                                content_checks[checkfile][i] = checkfn(content)
                            else:
                                content_checks[checkfile][i] = checkfn

                for i, filefn in enumerate(file_filter_fnlist):
                    if filefn(files):
                        file_checks[i]=True
        if check_all:
            if all(file_checks) and all([x for v in content_checks.values() for x in v]):
                filtered += [os.path.abspath(f)]
        else:
            if any(file_checks) and any([x for v in content_checks.values() for x in v]):  # Found a test dir
                filtered += [os.path.abspath(f)]
    return filtered


def filter_folders_model_arch(base='.', model_arch_filter='use_class_weights'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir
                if 'model_arch.txt' in files:
                    with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                        content = file.read()
                    checks.append(model_arch_filter in content)
        if len(checks) > 0 and any(checks):  # Found a test dir
            print(f"{f} Train session used {model_arch_filter} param.")
            filtered += [f]
    return filtered


def filter_folders_age(base='.', newer_than=1619628667):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files) == 0:
                continue
            if len(dirs) > 0:  # Traverse more
                continue
            if len(dirs) == 0 and len(files) > 0:  # leaf dir
                if 'params.txt' in files:
                    fname = pathlib.Path(os.path.join(root, 'params.txt'))
                    ctime = int(fname.stat().st_mtime)
                    checks.append(ctime > newer_than)
        if len(checks) > 0 and all(checks):  # Found a test dir
            print(f"{f} Train session is newer than {newer_than}.")
            filtered += [f]
    return filtered


def move_folders_to_old(base='.', folders=None):
    move_to_old_base = os.path.join(base, 'old')
    if folders is None:
        return
    for f in folders:
        move_to_old = os.path.join(move_to_old_base, *f.split(os.sep)[1:])
        print(f, move_to_old)
        move_filetree(f, move_to_old)


def create_symbolics(folders, symbolic_dir, symbolic_base_dir='../models_symbolic_links'):
    dst = os.path.join(symbolic_base_dir, symbolic_dir)
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in folders:
        sym_path = os.path.join(dst, os.path.split(f.rstrip(os.sep))[1])
        print(f"{f} to {sym_path}")
        try:
            os.symlink(os.path.abspath(f), os.path.abspath(sym_path))
        except FileExistsError as e:
            print(f"File already exists: {sym_path}\n{e}")


def rename_model_folders(base='.', folders=None, rename_in_test=False):
    folders = folders or glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        if "explain" in f:
            continue
        if rename_in_test or not ('test' in f):
            print(f"Checking {f}...")
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files) == 0:
                    continue
                if len(dirs) > 0:  # Traverse more
                    continue
                if len(dirs) == 0 and len(files) > 0:  # leaf dir (model?)
                    name = os.sep.join(root.split(os.sep)[:-1] + [root.split(os.sep)[-1].split('|')[0]])
                    is_cpc = True

                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()
                        if 'splits_file=' in content:
                            name += '|' + content.split("splits_file='")[1].split("'")[0].replace('.txt', '').replace(
                                '.', '_')
                        if 'use_class_weights=True' in content:
                            name += '|use_weights'

                    if 'model_arch.txt' in files:
                        with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                            content = file.read()
                        if 'StridedEncoder' in content:
                            name += '|strided'
                        if 'BaselineNet' in content:
                            is_cpc = False

                    if is_cpc:
                        if 'model_variables.txt' in files:
                            with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                                content = file.readlines()
                                for i, line in enumerate(content):
                                    if '{' in line:
                                        content = '\n'.join(content[i:])
                                        break
                            data = json.loads(content)
                            if '"freeze_cpc": false' in content:
                                name += '|unfrozen'
                            elif is_cpc:
                                name += '|frozen'
                            if '"use_context": true' in content:
                                name += '|C'
                            if '"use_latents": true' in content:
                                name += '|L'
                            if '"normalize_latents": true' in content:
                                name += '|LNorm'
                            if "sampling_mode" in content:
                                m = content.split('"sampling_mode": ')[1].split(',')[0][1:-1]
                                name += f'|m:{m}'
                            #if '"architectures_cpc.cpc_combined.CPCCombined":' in content:
                            # try:
                            #     cpc_type = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"].keys())[0]
                            #     name += "|"+cpc_type.split('.')[-2]
                            #     auto = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"][cpc_type]["_modules"][""]["autoregressive"])[0]
                            #     name += "|"+auto.split('.')[-2]
                            #     enc = list(data["architectures_cpc.cpc_combined.CPCCombined"]["_modules"][""]["cpc_model"][cpc_type]["_modules"][""]["encoder"])[0]
                            #     name += "|"+enc.split('.')[-2]
                            # except KeyError:
                            #     print("Model does not follow CPC Combined architecture spec.")
                            if '"downstream_model":' in content:
                                m = \
                                content.split('"downstream_model": {')[1].split('": {')[0].strip().lstrip('"').split(
                                    '.')[-2]
                                name += f'|{m}'

                        if 'params.txt' in files:
                            with open(os.path.join(root, 'params.txt'), 'r') as file:
                                content = file.read()
                            # if 'use_class_weights=True' in content:
                            #     name += '|use_weights'
                            # if 'downstream_epochs' in content:
                            #     epos = content.split('downstream_epochs=')[1].split(',')[0]
                            #     name += f'|dte:{epos}'
                            # if 'pretrain_epochs' in content and is_cpc:
                            #     epos = content.split('pretrain_epochs=')[1].split(',')[0]
                            #     name += f'|pte:{epos}'


                    else:  # not cpc

                        if 'model_variables.txt' in files:
                            with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                                content = file.readlines()
                                for i, line in enumerate(content):
                                    if '{' in line:
                                        content = '\n'.join(content[i:])
                                        break
                            data = json.loads(content)
                            name += '|ConvLyrs:' + str(count_key_in_dict(data, 'torch.nn.modules.conv.Conv1d'))
                            if 'torch.nn.modules.pooling.MaxPool1d' in content:
                                name += '|MaxPool'
                            if 'torch.nn.modules.pooling.AdaptiveAvgPool1d' in content:
                                name += '|AvgPool'
                            if 'torch.nn.modules.linear.Linear' in content:
                                name += '|Linear'
                            if 'torch.nn.modules.rnn.LSTM' in content:
                                name += '|LSTM'
                            if 'torch.nn.modules.batchnorm.BatchNorm1d' in content:
                                name += '|BatchNorm'
                            name += '|stride_sum:' + str(
                                int(np.array(extract_values_for_key_in_dict(data, 'stride')).sum()))
                            name += '|dilation_sum:' + str(
                                int(np.array(extract_values_for_key_in_dict(data, 'dilation')).sum()))
                            name += '|padding_sum:' + str(
                                int(np.array(extract_values_for_key_in_dict(data, 'padding')).sum()))
                            name += '|krnls_sum:' + str(
                                int(np.array(extract_values_for_key_in_dict(data, 'kernel_size')).sum()))

                    print(f"Renaming {root} to {name}")
                    try:
                        os.rename(root, name)
                    except OSError as ose:
                        if ose.errno == 36:
                            os.rename(root, name[0:254])
                        else:
                            raise


def long_to_abbreviation(name):
    parts = name.split('|')[0].split('.')
    shortname = parts[-2]
    shortname = shortname.replace('cpc_combined', 'cpc')
    shortname = shortname.replace('baseline', 'bl')

    return shortname


def is_test_folder(files):
    c1 = any([f.endswith('.csv') and f.startswith('labels-') for f in files])
    c2 = any([f.endswith('output.csv') for f in files])
    c3 = 'params.txt' in files and 'model_arch.txt' in files

    return c1 and c2 and c3


def is_train_folder(files):
    c1 = any([f.endswith('.pt') and 'checkpoint' in f for f in files])
    c2 = any([f.endswith('full_model.pt') for f in files])
    c3 = 'params.txt' in files and 'model_arch.txt' in files
    c4 = any([f.endswith('.pickle') for f in files])
    return c1 and c2 and c3 and c4


def write_models_to_dirs(base='.'):
    model_file = '../train-folder-models'
    arch_file = 'model_arch.txt'
    with open(model_file, 'r') as f:
        for line in f.readlines():
            mclass, path = line.split(':')
            mp = path.strip()[1:-1]
            ap = os.path.join(mp, arch_file)
            if not os.path.isfile(ap):
                with open(ap, 'w') as apf:
                    apf.write(mclass)


def clean_rename(folders=None):
    correct_age = set(filter_folders_age(newer_than=1619628667))  # Newer than introduction of correct train-test-split
    incorrect_age = set(filter_folders_age(newer_than=0)) - correct_age
    uses_weights_all = set(filter_folders_params(params_filter='use_class_weights=True'))
    uses_weights = (correct_age & uses_weights_all)
    print(uses_weights)
    uses_no_weights = correct_age - uses_weights
    print(uses_no_weights)
    rename_model_folders(folders=folders or correct_age)
    rename_folders_into_models(folders=folders or correct_age)
    # create_symbolics(uses_weights, 'class_weights')
    # create_symbolics(uses_no_weights, 'no_class_weights')
    # create_symbolics(train_folders & uses_weights, 'train/class_weights')
    # create_symbolics(train_folders & uses_no_weights, 'train/no_class_weights')


def clean_categorize(test=False):
    uses_model_variables = set(filter_folders_model_variables(model_variables_filter=''))
    correct_age = set(filter_folders_age(newer_than=1635760539)) & uses_model_variables #1619628667
    if not test:
        train_or_test_folders = set(filter(lambda x: not 'test' in x, correct_age))
    else:
        train_or_test_folders = set(filter(lambda x: 'test' in x, correct_age))
    baseline_folders = set(filter_folders_model_arch(model_arch_filter='BaselineNet')) & train_or_test_folders
    print(baseline_folders)
    cpc_folders = train_or_test_folders - baseline_folders
    correct_epochs_baseline_folders = set(
        filter_folders_params(params_filter='downstream_epochs=120')) & baseline_folders
    correct_epochs_cpc_folders = set(filter_folders_params(params_filter='downstream_epochs=20')) & cpc_folders
    uses_weights = set(filter_folders_params(params_filter='use_class_weights=True')) & correct_age
    splits = {}
    splits['min_cut-25'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits_min_cut25.txt'")) & correct_age
    splits['min_cut-50'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits_min_cut50.txt'")) & correct_age
    splits['min_cut-100'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits_min_cut100.txt'")) & correct_age
    splits['min_cut-150'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits_min_cut150.txt'")) & correct_age
    splits['min_cut-200'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits_min_cut200.txt'")) & correct_age

    splits['fewer_labels-0_01'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels0.01.txt'")) & correct_age
    splits['fewer_labels-0_05'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels0.05.txt'")) & correct_age
    splits['fewer_labels-0_001'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels0.001.txt'")) & correct_age
    splits['fewer_labels-0_005'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels0.005.txt'")) & correct_age
    splits['fewer_labels-10'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels10.txt'")) & correct_age
    splits['fewer_labels-14'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels14.txt'")) & correct_age
    splits['fewer_labels-20'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels20.txt'")) & correct_age
    splits['fewer_labels-30'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels30.txt'")) & correct_age
    splits['fewer_labels-40'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels40.txt'")) & correct_age
    splits['fewer_labels-50'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels50.txt'")) & correct_age
    splits['fewer_labels-60'] = set(
        filter_folders_params(params_filter="splits_file='train-test-splits-fewer-labels60.txt'")) & correct_age
    other_splits = correct_age - set(chain.from_iterable(splits.values()))

    uses_no_weights = correct_age - uses_weights
    base_dir = 'test' if test else 'train'
    for sfile, v in splits.items():
        create_symbolics(correct_epochs_cpc_folders & uses_weights & v,
                         base_dir + '/class_weights/' + '/few-labels/cpc/' + sfile)
        create_symbolics(correct_epochs_cpc_folders & uses_no_weights & v,
                         base_dir + '/no_class_weights/' + '/few-labels/cpc/' + sfile)
        create_symbolics(correct_epochs_baseline_folders & uses_weights & v,
                         base_dir + '/class_weights/' + '/few-labels/baseline/' + sfile)
        create_symbolics(correct_epochs_baseline_folders & uses_no_weights & v,
                         base_dir + '/no_class_weights/' + '/few-labels/baseline/' + sfile)
    create_symbolics(correct_epochs_cpc_folders & uses_weights & other_splits,
                     base_dir + '/class_weights/other-splits/cpc/')
    create_symbolics(correct_epochs_cpc_folders & uses_no_weights & other_splits,
                     base_dir + '/no_class_weights/other-splits/cpc/')
    create_symbolics(correct_epochs_baseline_folders & uses_weights & other_splits,
                     base_dir + '/class_weights/other-splits/baseline/')
    create_symbolics(correct_epochs_baseline_folders & uses_no_weights & other_splits,
                     base_dir + '/no_class_weights/other-splits/baseline/')


def clean_remove_dry_run():
    move_incomplete_training_folders()

def cut_off_attrs_from_path(tf):
    splits = os.path.normpath(tf).split(os.path.sep)
    newp = os.path.sep.join([s.split('|')[0] for s in splits])
    return newp.strip('.'+os.path.sep)

def move_paramstxt_to_test(base='.', folders=None):

    folders = folders or glob(os.path.join(base, "*_*_*", ""))
    train_folders = []
    for f in folders:
        if 'train' in f:
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files) == 0:
                    continue
                if len(dirs) > 0:  # Traverse more
                    continue
                if len(dirs) == 0 and len(files) > 0:  # leaf dir (model?)
                    if is_train_folder(files):
                        if 'params.txt' in files:
                            train_folders += [root]
    train_folders_cutoff = [cut_off_attrs_from_path(tf) for tf in train_folders]
    print(train_folders_cutoff)
    for f in folders:
        if 'test' in f:
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files) == 0:
                    continue
                if len(dirs) > 0:  # Traverse more
                    continue
                if len(dirs) == 0 and len(files) > 0:  # leaf dir (model?)
                    if is_test_folder(files):
                        # print("testing:")
                        # print(root)
                        # print(any([cut_off_attrs_from_path(root).endswith(tf) for tf in train_folders_cutoff]))
                        matching_train_folders = list(filter(lambda i_p: cut_off_attrs_from_path(root).endswith(i_p[1]), enumerate(train_folders_cutoff)))
                        # print(matching_train_folders)
                        if len(matching_train_folders) == 1:

                            i, p = matching_train_folders[0]
                            try:
                                train_params = os.path.join(train_folders[i], 'params.txt')
                                test_params = os.path.join(root, 'train_params.txt')
                                #print(f"copying {train_params}, to {test_params}")
                                shutil.copyfile(train_params, test_params)
                            except FileNotFoundError as e:
                                print(e)
                        elif len(matching_train_folders) > 1:
                            print("Found multiple matching folders!",root, matching_train_folders)
                        elif len(matching_train_folders) == 0:
                            # print("Didnt find folders for", root)
                            # print(cut_off_attrs_from_path(root))
                            pass



if __name__ == '__main__':
    # for i in range(10): #run this multiple time to remove nested folders
    #     print('Deletion routine:', i)
    #
    # rename_folders_into_test()
    # write_models_to_dirs()
    # rename_folders_into_models()
    # rename_folders_into_splits()
    # move_paramstxt_to_test()
    # move_folders_to_old(folders=incorrect_age)
    #
    #print(is_test_folder(glob('.')))
    clean_remove_dry_run()
    clean_rename()
    # # print(filter_folders_universal( #Find all models that normalize latents and have trained downstream
    # #     file_content_filter_fnlist_dict={'model_variables.txt': [lambda x: '"normalize_latents": true' in x], 'params.txt':[lambda x: 'use_class_weights=False' in x]}, file_filter_fnlist=[lambda x: any([y.endswith("_checkpoint_epoch_20.pt") for y in x])]))
    clean_remove_dry_run()
    clean_rename()

    # print(filter_folders_universal(
    #     folders=filter_folders_age(newer_than=1635779308), #CPC and tested
    #     file_content_filter_fnlist_dict={
    #         'model_variables.txt':[
    #             lambda x: "architectures_cpc.cpc_combined.CPCCombined" in x
    #         ]
    #     },
    #     file_filter_fnlist=[
    #         lambda f: 'labels-dataloader-0.csv' in f
    #     ]))
    #clean_categorize(test=True)
    # print(filter_folders_universal( #Find all models that normalize latents and have trained downstream
    #     file_content_filter_fnlist_dict={'model_variables.txt': [lambda x: '"normalize_latents": true' in x], 'params.txt':[lambda x: 'use_class_weights=False' in x]}, file_filter_fnlist=[lambda x: any([y.endswith("_checkpoint_epoch_20.pt") for y in x])]))
    # print(filter_folders_universal( #Find all models that normalize latents and have NOT trained downstream (and not test)
    #     file_content_filter_fnlist_dict={'model_variables.txt': [lambda x: '"normalize_latents": true' in x]}, file_filter_fnlist=[lambda x: not any([y.endswith("_checkpoint_epoch_20.pt") for y in x]), lambda x: not "labels-dataloader-0.csv" in x]))
    # print(torch.version.__version__)
    # cpc_folders = train_folders - baseline_folders
    # rename_folders_into_models(folders=['models/23_06_21-20-train|+(4x)cpc'])
    # rename_model_folders(folders=['/home/julian/Downloads/Github/contrastive-predictive-coding/models/11_08_21-15-58-test'], rename_in_test=True)
