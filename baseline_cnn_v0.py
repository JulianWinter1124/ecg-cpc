import torch
from torch import nn


class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes, verbose):
        super().__init__()
        self.n_out_classes = out_classes
        self.verbose = verbose
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=3),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_channels=9488, out_channels=1, kernel_size=1)

        self.fc = nn.Linear(out_channels, out_classes)
        #self.activation = nn.LogSoftmax(dim=1)
        #self.criterion = nn.NLLLoss()
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss() #nn.MultiLabelSoftMarginLoss()

    def forward(self, X, y=None):
        if self.verbose: print('input shape', X.shape)
        batch, window_size, channels = X.shape
        x = X.transpose(1, 2)
        if self.verbose: print(x.shape)
        x = self.convs(x)
        if self.verbose: print('x shape after conv', x.shape)
        x = x.transpose(1, 2)
        if self.verbose: print('x shape after transpose', x.shape)
        x = self.downsample(x)
        if self.verbose: print('x shape after downsample', x.shape)
        x = self.fc(x).squeeze(1)
        if self.verbose: print('x shape after fc', x.shape)
        logits = self.activation(x)
        if not y is None:
            #loss = self.criterion(logits, y)
            loss = torch.sum(torch.square(y-logits)) # Simple own implementation
            #loss = self.criterion(logits, torch.argmax(y, dim=1))
            accuracies = []
            mask = y != 0.0
            inverse_mask = ~mask
            zero_fit = 1.0 - torch.sum(torch.square(y[inverse_mask] - logits[inverse_mask])) / torch.sum(inverse_mask) #zero fit goal
            class_fit = 1.0 - torch.sum(torch.square(y[mask] - logits[mask])) / torch.sum(mask) #class fit goal
            accuracies.append(0.5*class_fit+0.5*zero_fit)
            accuracies.append(class_fit)
            accuracies.append(zero_fit)

            accuracies.append(1.0-torch.sum(torch.square(y-logits))/(self.n_out_classes*batch)) #Distance between all values
            #accuracy = 1.0 - torch.sum(torch.square(y - logits)) / torch.sum((y != 0.0) | (logits != 0.0)) #only count non zeros in accuracy?
            accuracies.append(torch.sum(torch.absolute(y-logits) <= 0.01)/(self.n_out_classes*batch)) #correct if probabilty within 0.01
            #accuracy = torch.sum(torch.eq(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))) / batch
            return accuracies, loss
        else:
            return logits

