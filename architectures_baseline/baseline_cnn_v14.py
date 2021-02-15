import torch
from torch import nn

from external.multi_scale_ori import MSResNet

class BaselineNet(nn.Module):

    def __init__(self, in_channels, out_channels, out_classes, verbose):
        super().__init__()
        self.n_out_classes = out_classes
        self.verbose = verbose
        self.msresnet = MSResNet(in_channels, num_classes=out_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x, y=None):
        if self.verbose: print('input shape', x.shape)
        batch, window_size, channels = x.shape
        x = x.transpose(1, 2) #torch.unsqueeze(torch.flatten(x, start_dim=1), 1)#.
        if self.verbose: print('after transpose', x.shape)
        fc_x, x  = self.msresnet(x)
        # if self.verbose: print('x shape after resnet', x.shape)
        # x = self.fc(x)
        # if self.verbose: print('x shape after fc', x.shape)
        logits = self.activation(fc_x)
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

