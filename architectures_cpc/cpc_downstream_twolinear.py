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
        if self.use_context and self.use_latents:
            self.rnn = nn.GRU(input_size=latent_size, hidden_size=context_size, num_layers=1, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(context_size*2, context_size*2),
                nn.ReLU(),
                nn.Linear(context_size*2, out_classes)
            )
        else: #if not both are getting used, normal classifier is enough for each
            self.classifier = nn.Sequential(
                nn.Linear(context_size, context_size),
                nn.ReLU(),
                nn.Linear(context_size, out_classes)
            )
        if self.use_latents:
            self.rnn = nn.GRU(input_size=latent_size, hidden_size=context_size, num_layers=1, batch_first=True)

        self.activation = nn.Sigmoid()


    def forward(self, latents=None, context=None, y=None):
        if self.verbose and latents: print('latents', latents.shape)
        if self.verbose and latents: print('context', context.shape)
        if self.use_context and self.use_latents:
            x, c = self.rnn(latents)
            context = torch.stack(context, x[:, -1])
            print(context.shape)
        elif self.use_latents:
            x, c = self.rnn(latents)
            context = x[:, -1]
        pred = self.classifier(context)
        output = self.activation(pred)
        #print('pred shape', pred.shape)
        if y is None:  #
            return output
        else: #Training mode return loss instead of prediction
            raise NotImplementedError