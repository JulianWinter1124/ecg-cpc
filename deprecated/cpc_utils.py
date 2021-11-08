import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# This class is not used right now


def info_NCE_loss(latents: torch.Tensor, context: torch.Tensor, future_latents: torch.Tensor, predictions: torch.Tensor,
                  target_dim=64, emb_scale=0):
    loss = 0.0
    latents = latents.permute(1, 0, 2)
    timesteps_in, batch_dim, n_latents = latents.shape
    timesteps_out, batch_dim, n_latents = predictions.shape

    total_elements = batch_dim * n_latents
    # print('Loss calculation.')
    # print('latents', latents.shape)
    # print('context', context.shape)
    # print('predictions', predictions.shape)
    # print('future_latents:', future_latents.shape)
    for i in range(timesteps_out):
        preds_i = predictions[i]
        logits = torch.mm(future_latents[i], torch.transpose(preds_i, 0, 1))
        labels = torch.arange(0, batch_dim).cuda()
        temp_loss = torch.sum(- labels * F.log_softmax(logits, -1),
                              -1)  # From https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
        mean_loss = temp_loss.mean()
        loss += mean_loss
    return loss


def info_NCE_loss_brian(latents: torch.Tensor, context, target_dim=64, emb_scale=0, steps_to_ignore=0,
                        steps_to_predict=3):
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
    # latents = latents.permute(1, 0, 2)
    targets = nn.Conv1d(in_channels=latents.shape[1], out_channels=target_dim, kernel_size=1).cuda()(latents)
    batch_dim, col_dim, row_dim, _ = targets.shape
    targets = targets.view([-1, target_dim])  # reshape
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
        temp_loss = torch.sum(- labels * F.log_softmax(logits, -1),
                              -1)  # From https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
        mean_loss = temp_loss.mean()
        loss += mean_loss
    return loss


def pixelCNN(latents):
    from keras.layers import Conv2D, ReLU
    from tensorflow_core.python import Pad
    import tensorflow as tf
    # latents: [B, H, W, D]
    cres = latents
    cres_dim = cres.shape[- 1]
    for _ in range(5):
        c = tf.nn.conv2d(filters=256, kernel_size=(1, 1))(cres)
        c = tf.nn.relu(c)
        c = tf.nn.conv2d(filters=256, kernel_size=(1, 3))(c)
        c = Pad(c, [[0, 0], [1, 0], [0, 0], [0, 0]])
        c = Conv2D(filters=256, kernel_size=(2, 1),
                   type='VALID')(c)
        c = ReLU(c)
        c = Conv2D(filters=cres_dim, kernel_size=(1, 1))(c)
        cres = cres + c
        cres = ReLU(cres)
    return cres


def CPC(latents, target_dim=64, emb_scale=0.1, steps_to_ignore=2, steps_to_predict=3):
    from tensorflow_core import reshape, matmul

    from keras.layers import Conv2D
    # latents: [B, H, W, D]
    loss = 0.0
    context = pixelCNN(latents)
    targets = Conv2D(output_channels=target_dim,
                     kernel_shape=(1, 1))(latents)
    batch_dim, col_dim, rows = targets.shape[: - 1]
    targets = reshape(targets, [- 1, target_dim])
    for i in range(steps_to_ignore, steps_to_predict):
        col_dim_i = col_dim - i - 1
        total_elements = batch_dim * col_dim_i * rows
        preds_i = Conv2D(output_channels=target_dim,
                         kernel_shape=(1, 1))(context)
        preds_i = preds_i[:, : - (i + 1), :, :] * emb_scale
        preds_i = reshape(preds_i, [- 1, target_dim])
        logits = matmul(preds_i, targets, transp_b=True)
        b = range(total_elements) / (col_dim_i * rows)
        col = range(total_elements) % (col_dim_i * rows)
        labels = b * col_dim * rows + (i + 1) * rows + col
        print(labels.shape)
    return loss

# if __name__ == '__main__':
#     B, H, W, D = (12, 4, 5, 100)
#     tf.compat.v1.enable_eager_execution()
#     latents = tensorflow.random.normal((B, H, W, D))
#     CPC(latents)
