import torch
from torch import nn


class BaselineNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False):
        super(BaselineNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(256, out_classes),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.features(x)
        x = torch.movedim(x, -1, 0)
        x, c_nh_n = self.lstm(x)  # no hidden state so its 0, take last output
        x = x[-1]
        x = self.classifier(x)
        x = self.activation(x)
        return x
