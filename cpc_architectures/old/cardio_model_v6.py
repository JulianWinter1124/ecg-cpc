
import torch
from torch import nn
from torch import functional as F
from cpc_utils import info_NCE_loss

class CPC(nn.Module):
    def __init__(self, encoder_model, autoregressive_model, predictor_model, timesteps_in, timesteps_out, timesteps_ignore=0):
        super(CPC, self).__init__()
        self.timesteps_in = timesteps_in
        self.timesteps_out = timesteps_out
        self.encoder = encoder_model
        self.autoregressive = autoregressive_model
        self.predictor = predictor_model
        #self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.cpc_train_mode = True

    def forward(self, x_windows, n_timesteps_in:int, n_timesteps_out:int, hidden, verbose=False):
        #TODO: make plot of data
        if verbose: print('x_windows has shape:', x_windows.shape) # shape = batch, windows, channels, window_size
        x_windows = x_windows.permute(1, 0, 2, 3) #reshaping into windows, batch, channels, windowsize
        n_windows, n_batches, _, _ = x_windows.shape
        if self.cpc_train_mode and n_windows != n_timesteps_in + n_timesteps_out:
            print("timesteps in and out not matching total windows")
        if verbose: print('x_windows has shape:', x_windows.shape)
        latent_list = []
        for x in x_windows[0:n_timesteps_in]:
            if verbose: print(x.shape)
            latent_list.append(self.encoder(x))
        latents = torch.stack(latent_list)
        if verbose: print('latents have shape:', latents.shape) # shape = timesteps,
        context, hidden = self.autoregressive(latents, hidden)
        context = context[-1, :, :] #We only need the latest state. Shape: batch, context_outputsize
        #TODO: decoder of latents
        #TODO: plot latents
        if not self.cpc_train_mode:
            return latents, context, hidden #CPC Mode
        #Calculate the loss
        loss = 0.0 #Will become Tensor
        correct = 0 #will become Tensor
        for k in range(0, n_timesteps_out): #Do this for each timestep
            latent_k = self.encoder(x_windows[-n_timesteps_out+k]) #batches, latents
            pred_k = self.predictor(context, k) # Shape (Batches, latents)
            softmax = self.lsoftmax(torch.mm(latent_k, pred_k.T)) #output: (Batches, Batches)
            correct += torch.sum(torch.eq(torch.argmax(softmax, dim=0), torch.arange(n_batches).cuda()))
            loss += torch.sum(torch.diag(softmax))
        loss /= n_batches * -1.0
        #loss /= (n_timesteps_out * n_batches * -1.0) #Unterschiedlich gewichten auch moeglich. Oder mehr als nur n_batches-1 negative samples
        #loss /= (n_timesteps_out * 1.0) #Man könnte auch jeden Timestep unterschiedlich gewichten. Nahe = wichtiger als entfernt
        accuracy = correct.true_divide(n_batches*n_timesteps_out)
        return accuracy, loss, hidden

    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def set_train_mode(self, mode_bool):
        self.cpc_train_mode = mode_bool

    def init_hidden(self, batch_size, use_gpu=True):
        return self.autoregressive.init_hidden(batch_size, use_gpu)



class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        filters = [10, 8, 4, 4, 4]  # See https://arxiv.org/pdf/1807.03748.pdf#Audio
        strides = [5, 4, 2, 2, 2]
        n_channels = [channels] + [latent_size] * len(filters)
        #self.batch_norm = nn.BatchNorm1d(n_channels[0]) #not used in paper?
        self.convolutionals = nn.Sequential(
            *[e for t in [
                (nn.Conv1d(in_channels=n_channels[i], out_channels=n_channels[i + 1], kernel_size=filters[i], stride=strides[i], padding=0),
                 #nn.BatchNorm1d(n_channels[i + 1]),
                 (nn.ReLU())
                 ) for i in range(len(filters))] for e in t]
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        def _weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x):
        # Input has shape (batches, channels, window_size)
        #print('Encoder input shape:', x.shape)
        #x = self.batch_norm(x)
        x = self.convolutionals(x)
        x = self.avg_pool(x)
        # Output has shape (batches, latents, 1)
        #Maybe squeeze??
        x = x.permute(2, 0, 1).squeeze(0) #Only squeeze first (NOT BATCH!) dimension
        #print('Encoder output shape:', x.shape)

        return x


class AutoRegressor(nn.Module):
    def __init__(self, n_latents, hidden_size):
        super(AutoRegressor, self).__init__()
        self.n_latents = n_latents
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=self.n_latents, hidden_size=self.hidden_size, num_layers=1)

        def _weights_init(m):
            for layer_p in self.gru._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, self.hidden_size).cuda()
        else: return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, x, hidden):
        #Input is (seq, batch, latents) maybe (13, 8, 128)
        #print('regressor input shape:', x.shape, hidden.shape)
        x, hidden = self.gru(x, hidden)
        #print('regressor output shape:', x.shape, hidden.shape)
        return x, hidden

class Predictor(nn.Module):
    def __init__(self, encoding_size, code_size, timesteps):
        super().__init__()
        self.code_size = code_size
        self.linears = nn.ModuleList(
            [nn.Linear(encoding_size, code_size)] * timesteps
        )
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.apply(_weights_init)


    def forward(self, x, timestep):
        #print('Predictor input shape:' , x.shape)
        prediction = self.linears[timestep](x)
        #print('Predictor output shape:', prediction.shape)
        return prediction
