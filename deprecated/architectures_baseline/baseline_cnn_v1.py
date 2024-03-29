import torch
from torch import nn


class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes, verbose=False):
        super().__init__()
        self.n_out_classes = out_classes
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=10, stride=5),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=8, stride=4),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=6, stride=3),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)  # TODO: FIX and try max

        self.fc = nn.Linear(in_channels, out_classes)
        # self.activation = nn.LogSoftmax(dim=1)
        # self.criterion = nn.NLLLoss()
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()  # nn.MultiLabelSoftMarginLoss()

    def forward(self, X, y=None):
        # print('input shape', X.shape)
        batch, window_size, channels = X.shape
        x = X.transpose(1, 2)
        # print(x.shape)
        x = self.convs(x)
        # print('x shape after convs', x.shape)
        x = self.pooling(x).squeeze(2)
        # print('x shape after pooling', x.shape)
        x = self.fc(x)
        # print('x shape after fc', x.shape)
        logits = self.activation(x)
        if not y is None:
            # loss = self.criterion(logits, y)
            loss = torch.sum(torch.square(y - logits))  # Simple own implementation
            # loss = self.criterion(logits, torch.argmax(y, dim=1))
            accuracies = []
            mask = y != 0.0
            inverse_mask = ~mask
            zero_fit = 1.0 - torch.sum(torch.square(y[inverse_mask] - logits[inverse_mask])) / torch.sum(
                inverse_mask)  # zero fit goal
            class_fit = 1.0 - torch.sum(torch.square(y[mask] - logits[mask])) / torch.sum(mask)  # class fit goal
            accuracies.append(0.5 * class_fit + 0.5 * zero_fit)
            accuracies.append(class_fit)
            accuracies.append(zero_fit)

            accuracies.append(
                1.0 - torch.sum(torch.square(y - logits)) / (self.n_out_classes * batch))  # Distance between all values
            # accuracy = 1.0 - torch.sum(torch.square(y - logits)) / torch.sum((y != 0.0) | (logits != 0.0)) #only count non zeros in accuracy?
            accuracies.append(torch.sum(torch.absolute(y - logits) <= 0.01) / (
                        self.n_out_classes * batch))  # correct if probabilty within 0.01
            # accuracy = torch.sum(torch.eq(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))) / batch
            return accuracies, loss
        else:
            return logits
