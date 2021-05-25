import torch
from torch import nn
#Time Series Classification from Scratch with Deep
#Neural Networks: A Strong Baseline https://arxiv.org/pdf/1611.06455.pdf
class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False) -> None:
        super(BaselineNet, self).__init__()
        self.verbose= verbose
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=out_classes, kernel_size=3),
            nn.BatchNorm1d(num_features=out_classes),
            nn.ReLU(),

        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if self.verbose: print(x.shape)
        x = x.transpose(1,2)
        if self.verbose: print(x.shape)
        x = self.features(x)
        if self.verbose: print(x.shape)
        x = torch.mean(x, dim=-1) #Global everage pooling
        if self.verbose: print(x.shape)
        x = self.activation(x)
        if self.verbose: print(x.shape)
        return x