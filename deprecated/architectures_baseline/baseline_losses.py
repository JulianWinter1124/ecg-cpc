from torch.nn import functional as F
import torch


def MSE_loss(pred, y):
    return F.mse_loss(pred, y)


def multi_loss(pred, y, loss_fns):
    return sum([lfn(pred, y) for lfn in loss_fns]) / len(loss_fns)
