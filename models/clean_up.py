import os
import pathlib
import shutil
from glob import glob
from pathlib import Path
from collections import Counter

from util.store_models import extract_model_files_from_dir


def move_incomplete_training_folders(base = '.'):
    print(f"Cleaning up.")
    move_to_base = os.path.join(base, 'delete')
    move_to_old_base = os.path.join(base, 'old')
    folders = glob(os.path.join(base, "*", ""))
    for f in folders:
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
                print("checking leaf", root)
                try:
                    param_file = os.path.join(root, 'params.txt')
                    #print(f"Looking for params.txt at {param_file}")
                    with open(param_file, 'r') as file:
                        content = file.read()
                        if not 'dry_run' in content:
                            print(f"Old model found (no dry_run param)")
                            move_filetree(root, move_to_old)
                        elif 'dry_run=True' in content:
                            #remove
                            print(f'Found dry_run=True. Removing {root}, {dirs}, {files}')
                            move_filetree(root, move_to)
                except:
                    print(f"Could not find params.txt in {root}")
                    move_filetree(root, move_to_old)

def move_filetree(source, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    print(f'Moving contents of {source} to {dest}')
    #remove
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
                if len(dirs) == 0 and len(files)==0:
                    continue
                if len(dirs) > 0: #Traverse more
                    continue
                if len(dirs) == 0 and len(files)>0: #leaf dir
                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()
                        checks.append('redo_splits' in content) #Not a test dir

        if len(checks) > 0 and all(checks): #Found a test dir
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
                if len(dirs) == 0 and len(files)==0:
                    continue
                if len(dirs) > 0: #Traverse more
                    continue
                if len(dirs) == 0 and len(files)>0: #leaf dir
                    checks.append(is_test_folder(files) and not is_train_folder(files)) #Not a test dir

        if len(checks) > 0 and all(checks): #Found a test dir
            print(f"{f} is a test directory. Renaming")
            os.rename(f, f.rstrip('/')+'-test(a)')

def rename_folders_into_models(base='.'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        model_names = []
        for root, dirs, files in os.walk(f):
            if len(dirs) > 0: #Do not look in leafs
                model_folders = list(filter(lambda x: '.' in x, dirs)) #model folders?
                print(f"Found the following model files: {model_folders} in {root}")
                model_names += map(long_to_abbreviation, model_folders)
        if len(model_names) > 0:
            counts = Counter(model_names)
            model_names_unique = sorted(list(map(lambda x: x[0] if x[1] <= 1 else f"({x[1]}x){x[0]}", counts.items())))
            old_folder_name = f.split('|')[0].rstrip('/')
            new_folder_name = old_folder_name + '|' + '+'.join(model_names_unique)
            print(f"Renaming {f} to {new_folder_name}")
            #os.rename(f, new_folder_name)

def filter_folders_params(base='.', params_filter='use_class_weights'):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files)==0:
                continue
            if len(dirs) > 0: #Traverse more
                continue
            if len(dirs) == 0 and len(files)>0: #leaf dir
                if 'params.txt' in files:
                    with open(os.path.join(root, 'params.txt'), 'r') as file:
                        content = file.read()
                    checks.append(params_filter in content)
        if len(checks) > 0 and any(checks): #Found a test dir
            print(f"{f} Train session used {params_filter} param.")
            filtered += [f]
    return filtered

def filter_folders_age(base='.', newer_than=1619628667):
    folders = glob(os.path.join(base, "*_*_*", ""))
    filtered = []
    for f in folders:
        checks = []
        for root, dirs, files in os.walk(f):
            if len(dirs) == 0 and len(files)==0:
                continue
            if len(dirs) > 0: #Traverse more
                continue
            if len(dirs) == 0 and len(files)>0: #leaf dir
                if 'params.txt' in files:
                    fname = pathlib.Path(os.path.join(root, 'params.txt'))
                    ctime = int(fname.stat().st_mtime)
                    checks.append(ctime>newer_than)
        if len(checks) > 0 and all(checks): #Found a test dir
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

def create_symbolics(folders, symbolic_dir, symbolic_base_dir = '../models_symbolic_links'):
    dst = os.path.join(symbolic_base_dir, symbolic_dir)
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in folders:
        sym_path = os.path.join(dst, os.path.split(f.rstrip(os.sep))[1])
        print(f"{f} to {sym_path}")
        try:
            os.symlink(os.path.abspath(f), os.path.abspath(sym_path))
        except FileExistsError as e:
            print(f"File already exists: {sym_path}\n{e}")


def rename_model_folders(base='.', folders = None):
    folders = folders or glob(os.path.join(base, "*_*_*", ""))
    for f in folders:
        if not ('test' in f):
            for root, dirs, files in os.walk(f):
                if len(dirs) == 0 and len(files)==0:
                    continue
                if len(dirs) > 0: #Traverse more
                    continue
                if len(dirs) == 0 and len(files)>0: #leaf dir (model?)
                    name = os.sep.join(root.split(os.sep)[:-1] + [root.split(os.sep)[-1].split('|')[0]])
                    is_cpc = True
                    if 'model_arch.txt' in files:
                        with open(os.path.join(root, 'model_arch.txt'), 'r') as file:
                            content = file.read()
                        if 'StridedEncoder' in content:
                            name += '|strided'
                        if 'BaselineNet' in content:
                            is_cpc = False

                    if 'model_variables.txt' in files:
                        with open(os.path.join(root, 'model_variables.txt'), 'r') as file:
                            content = file.read()
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
                        if '"downstream_model":' in content:
                            m = content.split('"downstream_model": {')[1].split('": {')[0].strip().lstrip('"').split('.')[-2]
                            name += f'|{m}'

                    if 'params.txt' in files:
                        with open(os.path.join(root, 'params.txt'), 'r') as file:
                            content = file.read()
                        if 'use_class_weights=True' in content:
                            name += '|use_weights'
                        if 'downstream_epochs' in content:
                            epos = content.split('downstream_epochs=')[1].split(',')[0]
                            name += f'|dte:{epos}'
                        if 'pretrain_epochs' in content and is_cpc:
                            epos = content.split('pretrain_epochs=')[1].split(',')[0]
                            name += f'|pte:{epos}'
                    print(f"Renaming {root} to {name}")
                    os.rename(root, name)


def long_to_abbreviation(name):
    parts = name.split('.')
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


def write_models_to_dirs(base = '.'):
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

if __name__ == '__main__':
    # for i in range(10): #run this multiple time to remove nested folders
    #     print('Deletion routine:', i)
    #move_incomplete_training_folders()
    #rename_folders_into_test()
    #write_models_to_dirs()
    #rename_folders_into_models()
    #rename_folders_into_splits()
    correct_age = set(filter_folders_age(newer_than=1619628667)) #Newer than introduction of correct train-test-split
    incorrect_age = set(filter_folders_age(newer_than=0)) - correct_age
    uses_weights_all = set(filter_folders_params(params_filter='use_class_weights=True'))
    uses_weights = (correct_age & uses_weights_all)
    print(uses_weights)
    uses_no_weights = correct_age - uses_weights
    print(uses_no_weights)
    rename_model_folders(folders=correct_age)
    #move_folders_to_old(folders=incorrect_age)
    #
    train_folders = set(filter(lambda x: not 'test' in x, correct_age))
    # create_symbolics(uses_weights, 'class_weights')
    # create_symbolics(uses_no_weights, 'no_class_weights')
    # create_symbolics(train_folders & uses_weights, 'train/class_weights')
    # create_symbolics(train_folders & uses_no_weights, 'train/no_class_weights')

