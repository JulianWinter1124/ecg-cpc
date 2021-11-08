from torch import nn


class PrintLayer(nn.Module):
    def __init__(self, name=''):
        super().__init__()
        self.name = name

    def forward(self, X):
        print(self.name, X.shape)
        return X
