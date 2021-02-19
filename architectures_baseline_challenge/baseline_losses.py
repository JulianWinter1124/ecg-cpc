from torch.nn import functional as F
import torch

def MSE_loss(pred, y):
    return F.mse_loss(pred, y)

def cross_entropy(pred, y):
    return torch.nansum(y*torch.log(pred))/y.shape[0]

def binary_cross_entropy(pred, y):
    return F.binary_cross_entropy(pred, y)


def multi_loss_function(loss_fns):
    def fn(pred, y):
        return sum([lfn(pred, y) for lfn in loss_fns]) / len(loss_fns)
    return fn