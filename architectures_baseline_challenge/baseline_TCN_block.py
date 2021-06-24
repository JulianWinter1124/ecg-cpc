import torch
from torch import nn
from torch.nn.utils import weight_norm

#might be broken (bad)
class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes, verbose):
        super().__init__()
        self.n_out_classes = out_classes
        self.verbose = verbose
        self.tcn_block = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0, dilation=1)),
            nn.ReLU(),
            nn.Dropout(0.2),

            weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=0, dilation=2)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.res_downsample = nn.Conv1d(in_channels=4500, out_channels=2245, kernel_size=1)
        #self.downsample = nn.Conv1d(in_channels=4745, out_channels=1, kernel_size=1)
        self.fc = nn.Linear(26940, out_classes)
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()
    def forward(self, x, y=None):
        if self.verbose: print('input shape', x.shape)
        batch, window_size, channels = x.shape
        x_res = x.clone()
        x = x.transpose(1, 2) #torch.unsqueeze(torch.flatten(x, start_dim=1), 1)#.
        if self.verbose: print('after transpose', x.shape)
        x = self.tcn_block(x)
        if self.verbose: print('x shape after tcn', x.shape)
        x_res = self.res_downsample(x_res)
        if self.verbose: print('x_res shape after residual downsample', x_res.shape)
        x = x.transpose(1, 2) + x_res
        if self.verbose: print('x shape after res add', x.shape)
        x = torch.flatten(x, start_dim=1)
        if self.verbose: print('x shape after flatten', x.shape)
        x = self.fc(x) #Like in https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py TODO: why is that okay?
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

