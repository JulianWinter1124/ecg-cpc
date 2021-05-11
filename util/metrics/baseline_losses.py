from torch.nn import functional as F
import torch
import numpy as np

def MSE_loss(pred, y):
    return F.mse_loss(pred, y)

def cross_entropy(pred, y):
    return -torch.sum(y*torch.log(pred))/np.prod(y.shape)

def bidirectional_cross_entropy(pred, y): #same as BCELoss
    pred_inv = 1.0-pred
    y_inv = 1.0-y
    return -(torch.nansum(y*torch.log(pred)) + torch.nansum(y_inv*torch.log(pred_inv)))/(np.prod(y.shape))

def binary_cross_entropy(pred, y):
    L = F.binary_cross_entropy(pred, y)
    return L

def multi_loss_function(loss_fns):
    def fn(pred, y):
        return sum([lfn(pred, y) for lfn in loss_fns]) / len(loss_fns)
    return fn

def nllloss(pred, y):
    F.nll_loss(pred, y)