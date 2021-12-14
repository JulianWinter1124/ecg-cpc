import torch
from torch import nn


class DownstreamLinearNet(nn.Module):

    def __init__(self, latent_size, context_size, out_classes, use_context=True, use_latents=False, verbose=False):
        super().__init__()
        self.latent_size = latent_size
        self.n_out_classes = out_classes
        self.use_context = use_context
        self.use_latents = use_latents
        self.verbose = verbose
        if self.use_context:
            self.classifier_c = nn.Sequential(
                nn.ReLU(),
                nn.Linear(context_size, context_size*2),
                nn.ReLU(),
                nn.Linear(context_size*2, out_classes)
            )
        if self.use_latents:
            self.classifier_l = nn.Sequential(
                nn.ReLU(),
                nn.Linear(latent_size, latent_size*2),
                nn.ReLU(),
                nn.Linear(latent_size*2, out_classes)
            )

        self.activation = nn.Sigmoid()

    def forward(self, latents=None, context=None, y=None):
        if self.verbose and latents: print('latents', latents.shape)
        if self.verbose and latents: print('context', context.shape)
        if self.use_context and self.use_latents:
            predc = self.classifier_c(context)
            predl = self.classifier_l(latents)
            predl = torch.max(predl, dim=0).values
            pred = (predc+predl)/2
        elif self.use_latents:
            pred = self.classifier_l(latents)
            pred = torch.max(pred, dim=0).values
        else:
            pred = self.classifier_c(context)
        output = self.activation(pred)  # do not squeeze on batch?
        # print('pred shape', pred.shape)
        if y is None:  #
            return output
        else:  # Training mode return loss instead of prediction
            raise NotImplementedError
