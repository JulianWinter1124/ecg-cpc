import os
import shutil
from glob import glob
from pathlib import Path

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
                try:
                    param_file = os.path.join(root, 'params.txt')
                    #print(f"Looking for params.txt at {param_file}")
                    with open(param_file, 'r') as f:
                        if not 'dry_run' in f.read():
                            print(f"Old model found (no dry_run param)")
                            move_filetree(root, move_to_old)
                        elif 'dry_run=False' in f.read():
                            #remove
                            print(f'Found dry_run=False. Removing {root}, {dirs}, {files}')
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
    rename_folders_into_test()
    #write_models_to_dirs()

