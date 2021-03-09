import torch
from torch import nn


class CPC(nn.Module):

    def __init__(self, encoder, autoregressive, predictor, timesteps_in, timesteps_out, latent_size, timesteps_ignore=0, normalize_latents=False, verbose=False):
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

        if not self.cpc_train_mode:
            return encoded_x, context[-1, :, :], hidden
        if self.verbose: print('context', context.shape)
        #offset = np.random.choice(encoded_x_steps - self.timesteps_out - self.timesteps_ignore, 10, replace=False)  # Draw 10 randoms
        loss = torch.tensor([0.0]).cuda()
        correct = torch.tensor([0.0]).cuda()
        assert len(context)-(self.timesteps_out+self.timesteps_ignore-self.timesteps_in) >= 1
        for i in range(self.timesteps_out):
            pred_latent = self.predictor(context[self.timesteps_in:-(self.timesteps_out+self.timesteps_ignore+1), :, :], i)
            encoded_latent = encoded_x[self.timesteps_in+i+1:-(self.timesteps_out+self.timesteps_ignore)+i, :, :].squeeze(0) #shape is batch, latent_size
            if self.verbose: print(pred_latent.shape, encoded_latent.shape)
            if self.normalize_latents:
                pred_latent /= torch.sqrt(torch.sum(torch.square(pred_latent)))
                encoded_latent /= torch.sqrt(torch.sum(torch.square(encoded_latent)))
            for step in range(pred_latent.shape[0]): #TODO: can this be broadcasted?
                softmax = self.lsoftmax(torch.mm(encoded_latent[step], pred_latent[step].T))  # output: (Batches, Batches)
                if self.verbose: print('softmax shape', softmax.shape)
                correct += torch.sum(torch.argmax(softmax, dim=0) == torch.arange(batch).cuda()) #Since
                loss += torch.sum(torch.diag(softmax))

        loss /= (batch * self.timesteps_out * pred_latent.shape[0]) * -1.0
        accuracy = correct.true_divide(batch * self.timesteps_out * pred_latent.shape[0])
        return accuracy, loss, hidden

    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True
