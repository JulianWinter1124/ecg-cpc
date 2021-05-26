import torch
from torch import nn
#Time Series Classification from Scratch with Deep
#Neural Networks: A Strong Baseline https://arxiv.org/pdf/1611.06455.pdf
class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes=94, verbose=False) -> None:
        super(BaselineNet, self).__init__()
        self.verbose = verbose
        
        self.conv1_1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=8, padding=4)
        self.bn1_1 = nn.BatchNorm1d(num_features=64)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn1_2 = nn.BatchNorm1d(num_features=64)
        self.conv1_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(num_features=64)
        
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, padding=4)
        self.bn2_1 = nn.BatchNorm1d(num_features=128)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn2_2 = nn.BatchNorm1d(num_features=128)
        self.conv2_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_3 = nn.BatchNorm1d(num_features=128)
        
        self.conv3_1 = nn.Conv1d(in_channels=128, out_channels=out_classes, kernel_size=8, padding=4)
        self.bn3_1 = nn.BatchNorm1d(num_features=out_classes)
        self.conv3_2 = nn.Conv1d(in_channels=out_classes, out_channels=out_classes, kernel_size=5, padding=2)
        self.bn3_2 = nn.BatchNorm1d(num_features=out_classes)
        self.conv3_3 = nn.Conv1d(in_channels=out_classes, out_channels=out_classes, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm1d(num_features=out_classes)
        
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x = x.transpose(1,2)
        print(x.shape)
        identity = x
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x + identity)
        print(x.shape)
        identity = x
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu(x + identity)
        print(x.shape)
        identity = x
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu(x + identity)
        print(x.shape)
        x = torch.mean(x, dim=-1)
        print(x.shape)
        x = self.activation(x)
        return x

