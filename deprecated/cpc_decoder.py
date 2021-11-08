from PIL import Image
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn


class CPCDecoder(nn.Module):

    def __init__(self, cpc_model_trained):
        super().__init__()
        self.cpc = cpc_model_trained
        self.cpc.set_train_mode(False)  # Makes the model not calculate loss/accuracy and not predict future latents
        self.cpc.freeze_layers()
        self.cpc.cuda()

    def forward(self, X, cpc_latents, cpc_contexts, cpc_hidden, y=None, finished=False):
        pred = None
        if not finished:
            cpc_latent, cpc_context, cpc_hidden = self.cpc(X, self.timesteps_in, -1, cpc_hidden)
            if self.show_latents:
                if self.counter % 10 == 0 and self.counter > 1000:
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
            # whole_context, hidden = self.gru(cpc_latents, hidden)
            # pred = self.gru_linear(whole_context[-1, :, :])
            stacked = torch.cat(cpc_latents, dim=0)  # outshape = (timesteps*m, batch,  latent_size)
            # print('stckd down', stacked.shape)
            summarized_X, hidden = self.gru(stacked)
            # print(summarized_X.shape)
            pred = self.linear(summarized_X[-1, :, :])

        pred = self.log_softmax(pred)
        # print('pred shape', pred.shape)
        if not y is None:  # Training mode return loss instead of prediction
            # print('y shape', y.shape)
            y = torch.argmax(y, 1)
            loss = self.criterion(pred, y)
            # print(torch.argmax(pred, dim=0), y)
            accuracy = torch.sum(torch.eq(torch.argmax(pred, dim=1), y)) / pred.shape[0]

            return accuracy, loss, cpc_hidden, (torch.argmax(pred, dim=1), y)
        else:
            return pred, cpc_hidden

    def init_hidden(self, batch_size, use_gpu):
        return self.cpc.init_hidden(batch_size, use_gpu)
