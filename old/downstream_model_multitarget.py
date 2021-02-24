from PIL import Image
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn


class DownstreamLinearNet(nn.Module):

    def __init__(self, cpc_model_trained, timesteps_in, context_size, latent_size, out_classes, use_context=False, use_latents=True):
        super().__init__()
        self.cpc = cpc_model_trained
        self.cpc.set_train_mode(False) #Makes the model not calculate loss/accuracy and not predict future latents
        #self.cpc.freeze_layers()
        self.cpc.cuda()
        self.use_context = use_context
        self.use_latents = use_latents
        self.hidden_size = 512
        if self.use_context and self.use_latents:
            self.linear = nn.Linear(in_features=latent_size+context_size, out_features=out_classes) #TODO: this is broken
        elif self.use_context:
            self.linear = nn.Linear(in_features=context_size, out_features=out_classes)
        else:  # If neither or just latents has been selected take latents
            #self.linear = nn.Linear(in_features=latent_size*timesteps_in, out_features=out_classes)

            self.gru = nn.GRU(input_size=latent_size, hidden_size=self.hidden_size, num_layers=1)
            self.linear = nn.Linear(in_features=self.hidden_size, out_features=out_classes)
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.show_latents = False
        self.counter = 0
        self.image_counter = 0
        self.timesteps_in = timesteps_in

    def forward(self, X, cpc_latents, cpc_contexts, cpc_hidden, y=None, finished=False):
        pred=None
        if not finished:
            cpc_latent, cpc_context, cpc_hidden = self.cpc(X, self.timesteps_in, -1, cpc_hidden)
            if self.show_latents:
                if self.counter % 10 == 0 and self.counter>1000:
                    copy = X.clone().detach().cpu().numpy()
                    for i in range(copy.shape[0]):
                        fig = plt.figure()
                        for c in range(copy.shape[2]):
                            plt.plot(copy[i, :, c, :].flatten())
                        plt.savefig("images/%d-%d-batch%d_signal.png" % (self.image_counter, self.counter, i))
                        plt.close(fig)
                    copy = cpc_latent.clone().detach().cpu().numpy()
                    for i in range(copy.shape[1]):
                        im = Image.fromarray(copy[:, i, :].T, 'L')
                        im.save("images/%d-%d-batch%d_latents.png" % (self.image_counter, self.counter, i))
                    self.image_counter += 1
            return cpc_latent, cpc_context, cpc_hidden
        self.image_counter = 0
        self.counter += 1

        if self.use_context and self.use_latents:
            conc = torch.cat([cpc_latents, cpc_contexts], dim=0)  # TODO: multi input nets... or concat?
            pred = self.linear(conc)

        elif self.use_context:  # use last context for now
            stacked = torch.cat(cpc_latents, dim=0)
            context, hidden = self.cpc.autoregressive(stacked, None)
            pred = self.linear(context[-1, :, :])

        else:  # use only latents, summarize and oredict with linear
            #whole_context, hidden = self.gru(cpc_latents, hidden)
            #pred = self.gru_linear(whole_context[-1, :, :])
            stacked = torch.cat(cpc_latents, dim=0) #outshape = (timesteps*m, batch,  latent_size)
            #print('stckd down', stacked.shape)
            summarized_X, hidden = self.gru(stacked)
            #print(summarized_X.shape)
            pred = self.linear(summarized_X[-1, :, :])

        pred = self.activation(pred)
        #print('pred shape', pred.shape)
        if not y is None:  # Training mode return loss instead of prediction
            batch, n_classes = y.shape
            #loss = self.criterion(pred, y)
            loss = torch.sum(torch.square(y - pred))
            accuracies = []
            mask = y != 0.0
            inverse_mask = ~mask
            zero_fit = 1.0 - torch.sum(torch.square(y[inverse_mask] - pred[inverse_mask])) / torch.sum(
                inverse_mask)  # zero fit goal
            class_fit = 1.0 - torch.sum(torch.square(y[mask] - pred[mask])) / torch.sum(mask)  # class fit goal
            accuracies.append(0.5 * class_fit + 0.5 * zero_fit)
            accuracies.append(class_fit)
            accuracies.append(zero_fit)

            accuracies.append(
                1.0 - torch.sum(torch.square(y - pred)) / (n_classes * batch))  # Distance between all values
            # accuracy = 1.0 - torch.sum(torch.square(y - logits)) / torch.sum((y != 0.0) | (logits != 0.0)) #only count non zeros in accuracy?
            accuracies.append(torch.sum(torch.absolute(y - pred) <= 0.01) / (
                        n_classes * batch))  # correct if probabilty within 0.01
            # accuracy = torch.sum(torch.eq(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))) / batch
            return accuracies, loss, cpc_hidden, (torch.argmax(pred, dim=1), y)
        else:
            return pred, cpc_hidden

    def init_hidden(self, batch_size, use_gpu):
        return self.cpc.init_hidden(batch_size, use_gpu)




