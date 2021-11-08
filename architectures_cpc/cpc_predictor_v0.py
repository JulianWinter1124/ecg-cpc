import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, context_size, latent_size, timesteps: int):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(context_size, latent_size)] * timesteps
        )

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x, timestep: int):
        # print('Predictor input shape:' , x.shape)

        if timestep is None:  # output shape = (timsteps, batch, latent_size)
            preds = []
            for i, l in enumerate(self.linears):
                preds.append(self.linears[i](x))
            return torch.stack(preds, dim=0)

        else:  # output shape = (batch, latent_size)
            return self.linears[timestep](x)
