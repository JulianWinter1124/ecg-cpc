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
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(in_channels=context_size*2, out_channels=out_classes, kernel_size=1, stride=1),
            )
        else: #if not both are getting used, normal classifier is enough for each (in_channels*1)
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(in_channels=context_size, out_channels=out_classes, kernel_size=1, stride=1),
            )
        if self.use_latents:
            self.rnn = nn.GRU(input_size=latent_size, hidden_size=context_size, num_layers=1, batch_first=False)

        self.activation = nn.Sigmoid()


    def forward(self, latents=None, context=None, y=None):
        if self.verbose and latents: print('latents', latents.shape)
        if self.verbose and latents: print('context', context.shape)
        if self.use_context and self.use_latents:
            x, c = self.rnn(latents)
            context = torch.cat([context, x[-1, :]], dim=-1)
            print(context.shape)
        elif self.use_latents:
            x, c = self.rnn(latents)
            context = x[:, -1]
        pred = self.classifier(context.unsqueeze(-1))
        output = self.activation(pred).squeeze()
        #print('pred shape', pred.shape)
        if y is None:  #
            return output
        else: #Training mode return loss instead of prediction
            raise NotImplementedError