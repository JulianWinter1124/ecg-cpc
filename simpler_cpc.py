import math

from torch import nn
import torch

from cpc_architectures import cpc_encoder_v1, cpc_encoder_v2, cpc_autoregressive_v0


class SimpleCPC(nn.Module):

    def __init__(self, encoder, decoder, autoregressive, timesteps_in, timesteps_out, latent_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.autoregressive = autoregressive
        self.timesteps_in = timesteps_in
        self.timesteps_out = timesteps_out
        self.latent_size = latent_size

    def forward(self, X, hidden=None):
        n_windows, batch, channels, length = X.shape
        assert n_windows >= self.timesteps_in + self.timesteps_out
        x = X.reshape((n_windows*batch, channels, length)) #squash into batch dimension
        encoded_x = self.encoder(x)
        encoded_x = encoded_x.reshape((n_windows, batch, *encoded_x.shape[1:])) #reshape into original
        if hidden is None:
            hidden = self.autoregressive.init_hidden(batch, use_gpu=False)
        predicted, hidden = self.autoregressive(encoded_x[:self.timesteps_in], hidden)
        print(predicted.shape)


    def downstream_forward(self, X, y):
        pass



if __name__ == '__main__':
    enc = cpc_encoder_v2.Encoder(3, 64)
    auto = cpc_autoregressive_v0.AutoRegressor(64, 64, layers=3)
    scpc = SimpleCPC(enc, None, auto, 4, 2, 64)
    X = torch.arange(6 * 12 * 3 * 100).float().reshape(6, 12, 3, 100)
    scpc(X)