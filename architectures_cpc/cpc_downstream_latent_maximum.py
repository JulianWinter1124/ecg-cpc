import torch
from torch import nn


class DownstreamLinearNet(nn.Module):

    def __init__(self, latent_size, context_size, out_classes, use_context=True, use_latents=False, verbose=False):
        super().__init__()
        self.latent_size = latent_size
        self.n_out_classes = out_classes
        self.use_context = False  # use_context
        self.use_latents = use_latents
        self.verbose = verbose
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, out_classes)
        )
        self.activation = nn.Sigmoid()

    def forward(self, latents=None, context=None, y=None):
        if self.verbose and latents: print('latents', latents.shape)  # steps, batch, ldim
        if self.verbose and latents: print('context', context.shape)
        x = self.classifier(latents)  # outshape, batch, steps, outshape
        x = torch.max(x, dim=0).values
        output = self.activation(x)
        # print('pred shape', pred.shape)
        if y is None:  #
            return output
        else:  # Training mode return loss instead of prediction
            raise NotImplementedError
