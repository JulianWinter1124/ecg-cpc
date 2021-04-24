import torch
from torch import nn
import numpy as np

from util.utility.sparse_print import SparsePrint


class CPC(nn.Module):

    def __init__(self, encoder, autoregressive, predictor, timesteps_in, timesteps_out, latent_size, timesteps_ignore=0, normalize_latents=False, verbose=False, sampling_mode='same'):
        super().__init__()
        self.encoder = encoder
        self.autoregressive = autoregressive
        self.predictor = predictor
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.timesteps_in = timesteps_in
        self.timesteps_out = timesteps_out
        self.timesteps_ignore = timesteps_ignore
        self.latent_size = latent_size
        self.normalize_latents = normalize_latents
        self.verbose = verbose
        self.cpc_train_mode = True
        if sampling_mode in ['all', 'same', 'random', 'future']:
            self.sampling_mode = sampling_mode
        else:
            self.sampling_mode = 'same'
        self.sprint = SparsePrint()

    def forward(self, X, y=None, hidden=None):
        if self.verbose: print('input', X.shape)
        if len(X.shape) == 4: #uses cpc or challenge dataset
            batch, windows, channels, length = X.shape
            X = X.transpose(1,2).reshape((batch, channels, -1))
        elif len(X.shape) == 3: #uses basline dataset, shape: (batch, length, channels)
            X = X.transpose(1, 2)
        batch, channels, length = X.shape #assume baseline dataset shape
        encoded_x = self.encoder(X) #encode whole data: shape: (batch, latent_size, length->out_length) #THIS IS NOT HOW ITS SHOWN IN THE PAPER
        encoded_x = encoded_x.permute(2, 0, 1)
        if self.verbose: print('encoder_x', encoded_x.shape)
        encoded_x_steps, _, _  = encoded_x.shape
        if hidden is None:
            hidden = self.autoregressive.init_hidden(batch_size=batch)
        context, hidden = self.autoregressive(encoded_x, hidden)
        if self.verbose: print('context', context.shape)

        if not self.cpc_train_mode:
            return encoded_x, context[-1, :, :], hidden
        #offset = np.random.choice(encoded_x_steps - self.timesteps_out - self.timesteps_ignore, 10, replace=False)  # Draw 10 randoms

        loss = torch.tensor([0.0]).cuda()
        correct = torch.tensor([0.0]).cuda()
        if encoded_x_steps-self.timesteps_out-self.timesteps_ignore-self.timesteps_in < 1:
            print('Not enough encoded values for CPC training. Lower timesteps_in/timesteps_out/timesteps_ignore or change window_size + encoder')
            return loss, correct, hidden
        if self.sampling_mode == 'same':
            for k in range(self.timesteps_out):
                pred_latent = self.predictor(context[self.timesteps_in:-(self.timesteps_out+self.timesteps_ignore+1), :, :], k)
                encoded_latent = encoded_x[self.timesteps_in+k+1:-(self.timesteps_out+self.timesteps_ignore)+k, :, :].squeeze(0) #shape is batch, latent_size
                if self.verbose: print(pred_latent.shape, encoded_latent.shape)
                if self.normalize_latents:
                    encoded_latent /= torch.norm(pred_latent, p=2, dim=-1, keepdim=True)
                    encoded_latent /= torch.norm(encoded_latent, p=2, dim=-1, keepdim=True)
                for step in range(pred_latent.shape[0]): #TODO: can this be broadcasted?
                    softmax = self.lsoftmax(torch.mm(encoded_latent[step], pred_latent[step].T))  # output: (Batches, Batches)
                    if self.verbose: print('softmax shape', softmax.shape)
                    correct += torch.sum(torch.argmax(softmax, dim=0) == torch.arange(batch).cuda()) #Since
                    loss += torch.sum(torch.diag(softmax))

            loss /= (batch * self.timesteps_out * pred_latent.shape[0]) * -1.0
            accuracy = correct.true_divide(batch * self.timesteps_out  * pred_latent.shape[0])
            return accuracy, loss, hidden

        if self.sampling_mode == 'all':
            encoded_latent = encoded_x.transpose(0, 1) #output shape: Batch, latents, latent_size
            #TODO: maybe also use all timesteps instead of random
            t = np.random.randint(low=self.timesteps_in, high=encoded_x_steps-self.timesteps_out-self.timesteps_ignore)  #Select current timestep at random

            if self.normalize_latents:
                encoded_latent = batch*encoded_latent/torch.norm(encoded_latent, p=2, dim=-1, keepdim=True)
            for k in range(self.timesteps_ignore, self.timesteps_out):
                pred_latent = self.predictor(context[t, :, :], k).squeeze(0)
                  # shape is batch, n_latents, latent_size
                if self.normalize_latents:
                    pred_latent = batch*pred_latent/torch.norm(pred_latent, p=2, dim=-1, keepdim=True) #Multiplying with batch so softmax stable
                    # prod = torch.matmul(encoded_latent, pred_latent.T).reshape(batch, encoded_x_steps*batch)
                    # prod_sum = torch.sum(prod, dim=-1, keepdim=True)
                    # softmax = torch.log(torch.div(prod, prod_sum))

                #torch.matmul(encoded_latent, pred_latent.T).shape == B x latents x B
                softmax = self.lsoftmax(torch.matmul(encoded_latent, pred_latent.T).reshape(batch, encoded_x_steps * batch))  # reshape for argmax to B, latents*B
                argmaxs = torch.argmax(softmax, dim=1)
                correct += torch.sum(argmaxs == torch.arange(batch).cuda()+t*batch) #Correct index will be at offset t
                loss += torch.sum(torch.diag(softmax.reshape(batch, encoded_x_steps, batch)[:, t+self.timesteps_ignore+k, :]))

            loss /= (batch * self.timesteps_out) * -1.0  # * pred_latent.shape[0]
            accuracy = correct.true_divide(batch * self.timesteps_out)  # * pred_latent.shape[0]
            return accuracy, loss, hidden

        if self.sampling_mode == 'random':
            raise NotImplementedError

        if self.sampling_mode == 'future':
            # TODO: maybe also use all timesteps instead of random
            t = np.random.randint(low=self.timesteps_in,
                                  high=encoded_x_steps - self.timesteps_out - self.timesteps_ignore)  # Select current timestep at random

            encoded_latent = encoded_x[t:].transpose(0, 1)  # output shape: Batch, latents*, latent_size
            encoded_x_steps = encoded_latent.shape[1]
            if self.normalize_latents:
                encoded_latent = batch*encoded_latent/torch.norm(encoded_latent, p=2, dim=-1, keepdim=True)
            for k in range(self.timesteps_ignore, self.timesteps_out):
                pred_latent = self.predictor(context[t, :, :], k).squeeze(0)
                # shape is batch, n_latents, latent_size
                if self.normalize_latents:
                    pred_latent = batch * pred_latent / torch.norm(pred_latent, p=2, dim=-1, keepdim=True)
                # torch.matmul(encoded_latent, pred_latent.T).shape == B x latents x B
                softmax = self.lsoftmax(torch.matmul(encoded_latent, pred_latent.T).reshape(batch,
                                                                                            encoded_x_steps * batch))  # reshape for argmax to B, latents*B
                argmaxs = torch.argmax(softmax, dim=1)
                correct += torch.sum(
                    argmaxs == torch.arange(batch).cuda())  # TODO: Correct index will be at offset t
                loss += torch.sum(torch.diag(softmax.reshape(batch, encoded_x_steps, batch)[:, k, :])) #+t is gone

            loss /= (batch * self.timesteps_out) * -1.0  # * pred_latent.shape[0]
            accuracy = correct.true_divide(batch * self.timesteps_out)  # * pred_latent.shape[0]
            return accuracy, loss, hidden

    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True
