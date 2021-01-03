import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

#This class is not used right now

def info_NCE_loss(latents: torch.Tensor, context:torch.Tensor, future_latents:torch.Tensor, predictions:torch.Tensor, target_dim=64 , emb_scale= 0):

    loss = 0.0
    latents = latents.permute(1, 0, 2)
    timesteps_in, batch_dim, n_latents = latents.shape
    timesteps_out, batch_dim, n_latents = predictions.shape

    total_elements = batch_dim * n_latents
    #print('Loss calculation.')
    #print('latents', latents.shape)
    #print('context', context.shape)
    #print('predictions', predictions.shape)
    #print('future_latents:', future_latents.shape)
    for i in range(timesteps_out):
        preds_i = predictions[i]
        logits = torch.mm(future_latents[i], torch.transpose(preds_i, 0, 1))
        labels = torch.arange(0, batch_dim).cuda()
        temp_loss = torch.sum(- labels * F.log_softmax(logits, -1), -1) #From https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
        mean_loss = temp_loss.mean()
        loss += mean_loss
    return loss

def info_NCE_loss_brian(latents: torch.Tensor, context, target_dim=64 , emb_scale= 0, steps_to_ignore=0 , steps_to_predict= 3):
    """
    Calculates the infoNCE loss for CPC according to 'DATA-EFFICIENT IMAGE RECOGNITION
    WITH CONTRASTIVE PREDICTIVE CODING' A.2
    :param latents: latent variables with shape [B, H, W, D]
    :param target_dim:
    :param emb_scale:
    :param steps_to_ignore:
    :param steps_to_predict:
    :return: The calculated loss
    """
    loss = 0.0
    #latents = latents.permute(1, 0, 2)
    targets = nn.Conv1d(in_channels=latents.shape[1], out_channels=target_dim, kernel_size=1).cuda()(latents)
    batch_dim, col_dim, row_dim, _ = targets.shape
    targets = targets.view([-1, target_dim])#reshape
    for i in range(steps_to_ignore, steps_to_predict):
        col_dim_i = col_dim - i - 1
        total_elements = batch_dim * col_dim_i * row_dim
        preds_i = nn.Conv1d(out_channels=target_dim, kernel_size=1)(context)
        preds_i = preds_i[:, : - (i + 1), :, :] * emb_scale
        preds_i = preds_i.view([- 1, target_dim])
        logits = torch.matmul(preds_i, torch.t(targets))
        b = np.arange(total_elements) / (col_dim_i * row_dim)
        col = np.arange(total_elements) % (col_dim_i * row_dim)
        labels = b * col_dim * row_dim + (i + 1) * row_dim + col
        temp_loss = torch.sum(- labels * F.log_softmax(logits, -1), -1) #From https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
        mean_loss = temp_loss.mean()
        loss += mean_loss
    return loss

