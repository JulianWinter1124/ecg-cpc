import torch
from torch import nn


class ExplainLabel(nn.Module):

    def __init__(self, baseline_model, normal_label_index=-1):
        super().__init__()
        self.model = baseline_model
        self.normal_label_index = normal_label_index

    def forward(self, X: torch.tensor, explain_class_index=-1, y=None):
        X.requires_grad = True
        if y is None:
            output = self.model(X)
            if explain_class_index > -1:
                fake_truth = torch.zeros_like(output)
                if self.normal_label_index>=0:
                    fake_truth[self.normal_label_index] = 1
            else:
                fake_truth = output.clone()
                fake_truth[:, explain_class_index] = 0.0
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



