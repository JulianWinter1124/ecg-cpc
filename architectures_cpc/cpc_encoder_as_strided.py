import torch
from torch import nn

class StridedEncoder(nn.Module):
    def __init__(self, cpc_encoder, window_size):
        super().__init__()
        self.window_layer = WindowLayer(window_size=window_size)
        self.cpc_encoder = cpc_encoder
        self.requires_grad_(True)

    def forward(self, X):
        X = self.window_layer(X)
        n_windows, n_batches, channels, window_size = X.shape
        latents = self.cpc_encoder(X.reshape(-1, *X.shape[2:]))  # reshape windows into batch dimension for only one forward
        latents = latents.reshape(n_windows, n_batches, *latents.shape[1:]).squeeze(-1) #shape is now n_windows, n_batches, n_latents
        latents = latents.movedim(0, -1) #shape is now n_batches, n_latents, steps
        return latents



class StridedWindowLayer(nn.Module):
    def __init__(self, window_size=10, stride=None, stack_dim=0):
        super().__init__()
        self.stride = window_size if stride is None else stride
        self.window_size = window_size
        self.stack_dim=0

    def forward(self, X):
        start_ix = torch.arange(0, X.shape[-1], self.stride)
        print(start_ix)
        return torch.narrow(X, dim=-1, start=start_ix, length=self.window_size)

class WindowLayer(nn.Module):
    def __init__(self, window_size=10, stack_dim=0):
        super().__init__()
        self.window_size = window_size
        self.stack_dim = 0

    def forward(self, X):
        in_shape = X.shape
        n_windows = X.shape[-1] // self.window_size #how many windows
        X = X.narrow(-1, 0, self.window_size*n_windows) #slice data
        X = X.view(*(X.shape[0:-1]), n_windows, self.window_size).movedim(-2, self.stack_dim)
        return X


if __name__ == '__main__':
    a = torch.stack([torch.stack([torch.arange(100)]*8)]*12, dim=1)
    o = WindowLayer(10)(a)
    StridedEncoder(None, 10)(a)