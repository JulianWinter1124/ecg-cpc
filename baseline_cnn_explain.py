import torch
from torch import nn, tensor


class ExplainLabel(nn.Module):

    def __init__(self, baseline_model):
        super().__init__()
        self.model = baseline_model

    def forward(self, X: torch.tensor, y=None):
        X.requires_grad = True
        if y is None:
            output = self.model(X)
            fake_truth = torch.zeros_like(output)
            loss = torch.sum(torch.square(fake_truth-output)) #Same as just sum of squared output in this case
            loss.backward()
            grad = torch.abs(X.grad)
            X.grad = None
            return output, grad
        else:
            accuracies, loss = self.model(X, y)
            loss.backward()
            grad = torch.abs(X.grad) #in Heat map distance is important not specific direction
            X.grad = None
            return accuracies, loss, grad



