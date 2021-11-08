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
            self.linear_context = nn.Linear(context_size, out_classes)
        if self.use_latents:
            self.output_size = 1
            self.pool = nn.AdaptiveMaxPool1d(output_size=self.output_size)
            self.linear_latents = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.output_size * latent_size, out_classes),
            )
        self.activation = nn.Sigmoid()
        # self.loss = nn.KLDivLoss()
        self.loss = nn.MSELoss()

    def forward(self, latents=None, context=None, y=None):
        if self.verbose and latents: print('latents', latents.shape)
        if self.verbose and latents: print('context', context.shape)
        if self.use_context and self.use_latents:  # some kind of ensemble?
            pred = (self.linear_context(context) + self.linear_latents(self.flatten(latents.transpose(0, 1)))) / 2.0
        elif self.use_context:
            pred = self.linear_context(context)
        elif self.use_latents:
            pooled = self.pool(latents.T).permute(1, 0,
                                                  -1)  # Reduce outsteps to outputsize and reshape to batch, latents, output_size
            pred = self.linear_latents(pooled)
        output = self.activation(pred)
        # print('pred shape', pred.shape)
        if y is None:  #
            return output
        else:  # Training mode return loss instead of prediction
            batch, n_classes = y.shape
            loss = self.loss(pred, y)
            accuracies = []
            mask = y != 0.0
            inverse_mask = ~mask
            zero_fit = 1.0 - torch.sum(torch.square(y[inverse_mask] - output[inverse_mask])) / torch.sum(
                inverse_mask)  # zero fit goal
            class_fit = 1.0 - torch.sum(torch.square(y[mask] - output[mask])) / torch.sum(mask)  # class fit goal
            accuracies.append(0.5 * class_fit + 0.5 * zero_fit)
            accuracies.append(class_fit)
            accuracies.append(zero_fit)

            accuracies.append(
                1.0 - torch.sum(torch.square(y - output)) / (self.n_out_classes * batch))  # Distance between all values
            # accuracy = 1.0 - torch.sum(torch.square(y - logits)) / torch.sum((y != 0.0) | (logits != 0.0)) #only count non zeros in accuracy?
            accuracies.append(torch.sum(torch.absolute(y - output) <= 0.01) / (
                    self.n_out_classes * batch))  # correct if probabilty within 0.01
            # accuracy = torch.sum(torch.eq(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))) / batch
            return accuracies, loss
