import torch
from torch import nn


class BaselineNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False):
        super(BaselineNet, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, out_classes),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x, c_nh_n = self.lstm(x)  # no hidden state so its 0, take last output
        x = x[:, -1]  # batch first so seq is at index 1
        x = self.classifier(x)
        x = self.activation(x)
        return x
