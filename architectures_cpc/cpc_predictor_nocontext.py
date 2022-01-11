import torch
from torch import nn

from external.tcn.TCN.tcn import TemporalConvNet


class Predictor(nn.Module):
    def __init__(self, context_size, latent_size, timesteps: int):
        super().__init__()
        self.tcn = TemporalConvNet(latent_size, [context_size, context_size, context_size, latent_size])

    def forward(self, x, timestep: int):
        # input shape = (timesteps, batch, latent_size)
        # print('Predictor input shape:' , x.shape)
        if timestep is None:  # output shape = (timesteps, batch, latent_size)
            return self.tcn(x.permute(1, 2, 0)).permute(2, 0, 1) #out (timesteps, batch, latent_size)
