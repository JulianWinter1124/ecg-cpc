import torch
from torch import nn


class CPCCombined(nn.Module):
    def __init__(self, cpc_model, downstream_model, freeze_cpc=True):
        super().__init__()
        self.cpc_model = cpc_model
        self.downstream_model = downstream_model
        self.freeze_cpc = freeze_cpc
        self.requires_grad_(True)
        self._unfreeze_cpc()


    def forward(self, X, y=None):
        print(self.freeze_cpc)
        if self.cpc_model.cpc_train_mode:
            self.cpc_model.cpc_train_mode=False
        if self.freeze_cpc:
            if not self.is_frozen:
                self._freeze_cpc()
            with torch.no_grad():
                encoded_x, context, hidden = self.cpc_model(X)
        else:
            if self.is_frozen:
                self._unfreeze_cpc()
            encoded_x, context, hidden = self.cpc_model(X)
        return self.downstream_model(encoded_x, context, y=y)

    def pretrain(self, X, y=None, hidden=None):
        if not self.cpc_model.cpc_train_mode:
            self.cpc_model.cpc_train_mode = True
        return self.cpc_model(X, y, hidden)

    def _freeze_cpc(self):
        self.is_frozen = True
        for param in self.cpc_model.parameters():
            param.requires_grad = False

    def _unfreeze_cpc(self):
        self.is_frozen = False
        for m in self.cpc_model.children():
            for param in m.parameters():
                param.requires_grad = True
        for param in self.cpc_model.parameters():
            param.requires_grad = True