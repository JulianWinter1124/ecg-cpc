from torch.nn import functional as F
import torch
import numpy as np

def MSE_loss(pred, y, weight=None):
    return F.mse_loss(pred, y, weight)

def cross_entropy(pred, y, weight=None):
    return -torch.sum(y*torch.log(pred))/np.prod(y.shape)

def bidirectional_cross_entropy(pred, y, weight=None): #same as BCELoss
    pred_inv = 1.0-pred
    y_inv = 1.0-y
    return -(torch.nansum(y*torch.log(pred)) + torch.nansum(y_inv*torch.log(pred_inv)))/(np.prod(y.shape))

def binary_cross_entropy(pred, y, weight=None):
    y.size()
    weight.size()
    return F.binary_cross_entropy(pred, y, weight=weight)

def multi_loss_function(loss_fns):
    def fn(pred, y, weight=None):
        return sum([lfn(pred, y, weight) for lfn in loss_fns]) / len(loss_fns)
    return fn

def nllloss(pred, y, weight=None):
    F.nll_loss(pred, y, weight=weight)