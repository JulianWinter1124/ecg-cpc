import torch
from torch import nn


class AutoRegressor(nn.Module):
    def __init__(self, n_latents, hidden_size, layers=1):
        super(AutoRegressor, self).__init__()
        self.n_latents = n_latents
        self.hidden_size = hidden_size
        self.layers = layers
        self.gru = nn.GRU(input_size=self.n_latents, hidden_size=self.hidden_size, num_layers=self.layers)

        def _weights_init(m):
            for layer_p in self.gru._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(self.layers, batch_size, self.hidden_size).cuda()
        else:
            return torch.zeros(self.layers, batch_size, self.hidden_size).cpu()

    def forward(self, x, hidden):
        # Input is (seq, batch, latents) maybe (13, 8, 128)
        # print('regressor input shape:', x.shape, hidden.shape)
        x, hidden = self.gru(x, hidden)
        # print('regressor output shape:', x.shape, hidden.shape)
        return x, hidden
