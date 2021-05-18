import glob
import json
import os
import pickle

import torch

from util.full_class_name import fullname


def save_model_checkpoint(output_path, epoch, model, optimizer=None, name=""):
    print("saving model at epoch:", epoch)
    checkpoint = {
        'epoch':epoch,
        'model_state_dict': model.state_dict()
    }
    if not optimizer is None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, os.path.join(output_path, name+'_checkpoint_epoch_{}.pt'.format(epoch)))

def save_model_architecture(output_path, model, name=""):
    print("Saving full model...")
    torch.save(model, os.path.join(output_path, name+'_full_model.pt'))
    save_model_architecture_text_only(output_path, model, name)

def save_model_architecture_text_only(output_path, model, name=""):
    with open(os.path.join(output_path, 'model_arch.txt'), 'w') as f:
        print(name, file=f)
        print(fullname(model), file=f)
        print(model, file=f)

def save_model_variables_text_only(output_path, model, name=""):
    with open(os.path.join(output_path, 'model_variables.txt'), 'w') as f:
        print(name, file=f)
        print(fullname(model), file=f)
        print(json.dumps(extract_params_from_model(model), sort_keys=True, indent=2), file=f)

def load_model_architecture(full_model_file, device_id='cuda:0'): #
    model = torch.load(full_model_file, map_location=device_id)
    return model

def load_model_checkpoint(model_checkpoint_file, model, optimizer=None, device_id='cuda:0'):
    checkpoint = torch.load(model_checkpoint_file, map_location=device_id)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.eval()
    return model, optimizer, epoch

def load_model(full_model_file, model_checkpoint_file, optimizer):
    model = load_model_architecture(full_model_file)
    model, optimizer, epoch = load_model_checkpoint(model_checkpoint_file, model, optimizer)
    return model, optimizer, epoch


def extract_model_files_from_dir(directory):
    files = []
    for root, dirs, dir_files in os.walk(directory):
        fm_temp, ch_temp = [], []
        for file in dir_files:
            if 'full_model' in file and file.endswith('.pt'):
                fm_temp.append(os.path.join(root, file))
            elif 'checkpoint' in file and file.endswith('.pt'):
                ch_temp.append(os.path.join(root, file))
        if len(fm_temp) > 0 and len(ch_temp) > 0:
            files.append((fm_temp, ch_temp))
    return files

def extract_params_from_model(obj, prefix=''):
    if issubclass(type(obj), torch.nn.Module):
        return extract_params_from_model(obj.__dict__, fullname(obj))
    elif issubclass(type(obj), dict):
        params = {}
        for k, v in obj.items():
            if not k.startswith('_') or k == '_modules':
                params.update({k:extract_params_from_model(v)})
        return {prefix:params} if params else {}
    elif issubclass(type(obj), list):
        return [extract_params_from_model(l) for l in obj]
    elif not hasattr(obj, '__dict__'):
        return obj
    else:
        return None


@DeprecationWarning
def save_model_state(output_path, epoch, name=None, model=None, optimizer=None, accuracies=None, losses=None, full=False):
    if name is None:
        name = fullname(model)
    with open(os.path.join(output_path, 'model_arch.txt'), 'w') as f:
        print(fullname(model), file=f)
        print(model, file=f)
    if full:
        print("Saving full model...")
        name = 'model_full.pt'
        torch.save(model, os.path.join(output_path, name))
        with open(os.path.join(output_path, 'model_arch.txt'), 'w') as f:
            print(fullname(model), file=f)
            print(model, file=f)
    else:
        print("saving model at epoch:", epoch)
        if not (model is None and optimizer is None):
            name = name + '_modelstate_epoch' + str(epoch) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(output_path, name))
        if not (accuracies is None and losses is None):
            with open(os.path.join(output_path, 'losses.pkl'), 'wb') as pickle_file:
                pickle.dump(losses, pickle_file)
            with open(os.path.join(output_path, 'accuracies.pkl'), 'wb') as pickle_file:
                pickle.dump(accuracies, pickle_file)

@DeprecationWarning
def load_model_state(model_path, model=None, optimizer=None):
    if model is None:
        model = torch.load(model_path)
        epoch = 1
    else:
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if not optimizer is None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
            epoch = 1

    model.eval()
    return model, optimizer, epoch