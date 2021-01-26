import torch
from torch import nn
import torch.nn.functional as F


class BaselineNet(nn.Module):

    def __init__(self, in_channels, latent_size, out_classes):
        super().__init__()
        self.n_out_classes = out_classes
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=latent_size, kernel_size=5, stride=4),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=4, stride=3),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=2),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=2),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=2, stride=1),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.t_convs = nn.Sequential(
            nn.ConvTranspose1d(in_channels=latent_size, out_channels=latent_size, kernel_size=2, stride=1, output_padding=0),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=latent_size, out_channels=latent_size, kernel_size=4, stride=3, output_padding=0),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=latent_size, out_channels=in_channels, kernel_size=5, stride=4, output_padding=3),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.fc = nn.Linear(latent_size, out_classes)
        #self.activation = nn.LogSoftmax(dim=1)
        #self.criterion = nn.NLLLoss()
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss() #nn.MultiLabelSoftMarginLoss()

    def forward(self, X, y=None):
        X = X.squeeze(1)
        #print('input shape', X.shape)
        batch, window_size, channels = X.shape
        #X = X.transpose(1, 2)
        x = X.clone()
        #print('x original shape', x.shape)
        x = self.convs(x)
        #print('x shape after convs', x.shape)
        unpool_dim = x.shape[-1]
        x = self.pooling(x)
        print('x shape after pooling', x.shape)

        if not y is None:
            x = x.squeeze(-1)
            x = self.fc(x)
            #print('x shape after fc', x.shape)
            logits = self.activation(x)

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
            reps = [1] * len(x.shape)
            reps[-1] = unpool_dim
            x = x.repeat(*reps)  # unpool avg layer
            x = F.conv_transpose1d(x)
            #print('x shape after unpool', x.shape)
            x = self.t_convs(x)
            #print('x shape after t_convs', x.shape)
            loss = torch.sum(torch.square(X-x)) #SE
            accuracies = []
            accuracies.append(torch.tensor(0.0).cuda())
            return accuracies, loss

