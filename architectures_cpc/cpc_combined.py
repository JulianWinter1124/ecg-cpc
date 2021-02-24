import torch
from torch import nn


class CPCCombined(nn.Module):
    def __init__(self, cpc_model, downstream_model, freeze_cpc=True):
        super().__init__()
        self.cpc_model = cpc_model
        self.downstream_model = downstream_model
        self.freeze_cpc = freeze_cpc

    def forward(self, X, y=None):
        if self.cpc_model.train_mode:
            self.cpc_model.train_mode=False
        if self.freeze_cpc:
            with torch.no_grad():
                encoded_x, context, hidden = self.cpc_model(X)
        else:
            encoded_x, context, hidden = self.cpc_model(X)
        return self.downstream_model(encoded_x, context, y=y)

    def pretrain(self, X, y=None, hidden=None):
        if not self.cpc_model.train_mode:
            self.cpc_model.train_mode = True
        return self.cpc_model(X, y, hidden)