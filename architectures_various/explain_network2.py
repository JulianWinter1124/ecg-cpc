import torch
from torch import nn, autograd
from torch.optim import Adam
from util.metrics import training_metrics, baseline_losses as bl


class ExplainLabel(nn.Module):
    def __init__(self, model, class_weights=None, cuda=True):
        super().__init__()
        self.model = model
        self.class_weights = class_weights

    def forward(self, X1:torch.tensor, y=None):
        if y is None:
            return self.model(X1)
            # output = self.model(X)
            # if explain_class_index > -1:
            #     fake_truth = torch.zeros_like(output)
            #     if self.normal_label_index>=0:
            #         fake_truth[self.normal_label_index] = 1
            # else:
            #     fake_truth = output.clone()
            #     fake_truth[:, explain_class_index] = 0.0
            # loss = torch.sum(torch.square(fake_truth-output)) #Same as just sum of squared output in this case
            # loss.backward()
            # grad = torch.abs(X.grad)
            # X.grad = None
            # return output, grad
        else:
            grads = []
            X = torch.tensor(X1, requires_grad=True, device=X1.device) # X1.clone().detach().requires_grad_(True)
            X.retain_grad()
            pred = self.model(X, y=None) # makes model return prediction instead of loss
            if len(pred.shape) == 1: #hack for squeezed batch dimension
                pred = pred.unsqueeze(0)
            loss = bl.binary_cross_entropy(pred=pred, y=y, weight=self.class_weights) #bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
            loss.backward()
            #grad = autograd.grad(loss, X, create_graph=True, retain_graph=True, allow_unused=True)#
            #print(grad)
            #print('################################3')
            #print(X)
            grad = X.grad
            #grad = torch.abs(grad) #in Heat map distance is important not specific direction
            X.grad=None
            return pred, grad
