import os

import numpy as np
import torch


class DataStorage():
    def __init__(self, save_path):
        self.data_dict = {}
        self.save_path = save_path

    def append(self, key:str, data:object):
        if key in self.data_dict:
            v = self.data_dict[key]
            if isinstance(data, torch.Tensor):
                self.data_dict[key] = torch.cat([v, data], dim=0)
        else:
            self.data_dict[key] = data

    def add(self, key:str, data:object):
        if key in self.data_dict:
            v = self.data_dict[key]
            if isinstance(data, torch.Tensor):
                self.data_dict[key] += data
            else:
                self.data_dict[key] += data
        else:
            self.data_dict[key] = data

    def save_to_file(self):
        for k, v in self.data_dict.items():
            p = os.path.join(self.save_path, k) + '.pt'
            if isinstance(v, torch.Tensor):
                torch.save(v, p)
            elif isinstance(v, np.Array):
                np.save(p, v)

def load_np_array(p):
    return np.load(p)

def load_torch_tensor(p):
    return torch.load(p)