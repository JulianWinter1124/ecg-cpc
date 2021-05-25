import torch
from torch import nn
#Time Series Classification from Scratch with Deep
#Neural Networks: A Strong Baseline https://arxiv.org/pdf/1611.06455.pdf
class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False) -> None:
        super(BaselineNet, self).__init__()
        self.verbose = verbose
        self.features = nn.Sequential(
            ResidualBlock(in_channels, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=out_classes)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = self.classifier(x)
        print(x.shape)
        x = torch.mean(x, dim=-1)

        print(x.shape)
        x = self.activation(x)
        if self.verbose: print(x.shape)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm1d(num_features=out_channels)
        ),
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        print(x.shape)
        x = self.block(x)
        print(x.shape)
        #x = x + identity
        print(x.shape)
        x = self.relu(x)
        return x