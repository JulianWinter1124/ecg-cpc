import torch
import cv2
from torch import nn, autograd
from torch.optim import Adam
from util.metrics import training_metrics, baseline_losses as bl
import functools
import numpy as np


class ExplainLabel(nn.Module):
    def __init__(self, model, class_weights=None, cuda=True):
        super().__init__()
        self.model = model
        self.class_weights = class_weights

    def forward(self, X1: torch.tensor, y=None):
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
            X = torch.tensor(X1, requires_grad=True, device=X1.device)  # X1.clone().detach().requires_grad_(True)
            X.retain_grad()
            pred = self.model(X, y=None)  # makes model return prediction instead of loss
            if len(pred.shape) == 1:  # hack for squeezed batch dimension
                pred = pred.unsqueeze(0)
            loss = bl.binary_cross_entropy(pred=pred, y=y,
                                           weight=self.class_weights)  # bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
            loss.backward()
            # grad = autograd.grad(loss, X, create_graph=True, retain_graph=True, allow_unused=True)#
            # print(grad)
            # print('################################3')
            # print(X)
            grad = X.grad
            # grad = torch.abs(grad) #in Heat map distance is important not specific direction
            X.grad = None
            return pred, grad

class ExplainLabelLayer(nn.Module):
    def __init__(self, model, class_weights=None, cuda=True, layer=None, guided=False):
        super().__init__()
        self.model = model
        if guided:
            convert_relu_to_custom_relu(self.model)
        self.class_weights = class_weights
        if type(layer) == str:
            self.layer = functools.reduce(lambda o, n: getattr(o, n), [self.model] + layer.split('.'))
        else:
            self.layer = layer



        self.guided = guided

        self.gradients = None
        self.data_gradient = None
        self.activations = None
        self._register_hooks()


    def _register_hooks(self):
        self.layer.register_forward_hook(self.save_activation)
        self.layer.register_full_backward_hook(self.save_gradient)
        # if not self.input_layer is None:
        #     self.input_layer.register_full_backward_hook(self.save_data_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.cpu().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].cpu().detach() #last layer comes first
    #
    # def save_data_gradient(self, module, grad_input, grad_output):
    #     self.data_gradient = grad_output[0].cpu().detach()

    def forward(self, X1: torch.tensor, y=None):
        self.gradients = None
        self.activations = None
        if self.guided:
            X1.requires_grad = True
            X1.grad = None
        self.data_gradient = None
        if y is None:
            return self.model(X1)
        else:

            pred = self.model(X1, y=None)  # makes model return prediction instead of loss
            if len(pred.shape) == 1:  # hack for squeezed batch dimension
                pred = pred.unsqueeze(0)
            loss = bl.binary_cross_entropy(pred=pred, y=y,
                                           weight=self.class_weights)  # bl.multi_loss_function([bl.binary_cross_entropy, bl.MSE_loss])(pred=pred, y=labels)
            loss.backward()
            if self.guided:
                self.data_gradient = X1.grad.cpu().detach()
            # grad = torch.abs(grad) #in Heat map distance is important not specific direction
            return pred

    def get_gradcam_weights(self):
        return torch.mean(self.gradients, dim=-1, keepdim=True) #over data dimension

    def get_gradcam(self, target_size=[1, 4500], scale=False):
        w = self.get_gradcam_weights()
        B, C, _ = w.shape
        x = torch.sum(w * self.activations, dim=1) #Sum channel dim
        print('cam shape', x.shape)
        x[x<0]=0
        if scale:
            x = x - x.min()
            x = x / (1e-12 + x.max())
        if target_size is None:
            return x
        else:
            return torch.Tensor(scale_cam_image(x, target_size=target_size))

    def get_gradcam_and_guided(self, target_size=[1, 4500]):
        print('data shape', self.data_gradient.shape)
        #self.data_gradient[self.data_gradient<0]=0.
        self.data_gradient = torch.abs(self.data_gradient)
        grad_img = torch.Tensor(scale_cam_image(self.data_gradient, target_size))
        print(grad_img.shape)
        B, L, C = grad_img.shape
        target_size = [C, L]
        cam = self.get_gradcam(target_size)
        print(cam.shape)
        return grad_img * cam


class F_GuidedRelu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = (grad_output>0).float() * (input>0).float() * grad_output
            return grad_input


class GuidedRelu(nn.Module):
    def __init__(self):
        super(GuidedRelu, self).__init__()
        self.relu = F_GuidedRelu.apply
    def forward(self, x):
        return self.relu(x)

def scale_cam_image(cam, target_size=None): #@https://github.com/jacobgil/pytorch-grad-cam/blob/770f29027598a8bf3ef660e04d14c44770b4d03c/pytorch_grad_cam/base_cam.py#L136
    result = []
    for img in cam:
        img = img - torch.min(img)
        img = img / (1e-12 + torch.max(img))
        if target_size is not None:
            img = cv2.resize(img.numpy(), target_size)
        result.append(img)
        print('img; ', img.shape)
    result = np.stack(result)
    print('RES; ', result.shape)
    return result


def convert_relu_to_custom_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, GuidedRelu())
        else:
            convert_relu_to_custom_relu(child)
