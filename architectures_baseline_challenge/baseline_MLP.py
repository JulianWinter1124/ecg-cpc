import torch
from torch import nn


# Time Series Classification from Scratch with Deep
# Neural Networks: A Strong Baseline https://arxiv.org/pdf/1611.06455.pdf
class BaselineNet(nn.Module):

    def __init__(self, in_features=4500, out_classes=94, verbose=False) -> None:
        super(BaselineNet, self).__init__()
        self.verbose = verbose
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=in_features, out_features=500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=500, out_features=out_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = x[:, :, 0]  # Only take first channel for this network
        x = self.classifier(x)
        x = self.activation(x)
        if self.verbose: print(x.shape)
        return x
