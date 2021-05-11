import torch
from torch import nn

class BaselineNet(nn.Module): #strides and kernel_sizes of alexnet to power of 2

    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False) -> None:
        super(BaselineNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11**2, stride=3**2, padding=2), #here 3 instead of 4
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3**2, stride=2**2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3**2, stride=2**2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3**2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3**2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3**2, stride=2**2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_classes),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.activation(x)
        return x