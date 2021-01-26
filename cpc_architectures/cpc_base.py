
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from cpc_utils import info_NCE_loss

class CPC(nn.Module):
    def __init__(self, encoder_model, autoregressive_model, predictor_model, latent_size, timesteps_in, timesteps_out, timesteps_ignore=0, verbose=False):
        super(CPC, self).__init__()
        self.timesteps_in = timesteps_in
        self.timesteps_out = timesteps_out
        self.latent_size = latent_size
        self.encoder = encoder_model
        self.autoregressive = autoregressive_model
        self.predictor = predictor_model
        #self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.cpc_train_mode = True

        self.verbose = verbose

    def forward(self, x_windows, n_timesteps_in:int, n_timesteps_out:int, hidden):
        if self.verbose: print('x_windows has shape:', x_windows.shape) # shape = batch, windows, channels, window_size
        x_windows = x_windows.transpose(1, 0) #reshaping into windows, batch, channels, windowsize
        n_windows, n_batches, _, _ = x_windows.shape
        if self.verbose: print('x_windows has shape:', x_windows.shape)
        if self.cpc_train_mode and n_windows != n_timesteps_in + n_timesteps_out:
            print("timesteps in and out not matching total windows")
        latents = self.encoder(x_windows.reshape(-1, *x_windows.shape[2:])) #reshape windows into batch dimension for only one forward
        latents = latents.reshape(n_windows, n_batches, *latents.shape[1:]) #reshape shape back
        if self.verbose: print('latents have shape:', latents.shape) # shape = windows, batch, latent_size
        context, hidden = self.autoregressive(latents[0:n_timesteps_in, :], hidden)
        context = context[-1, :, :] #We only need the last state. Shape: batch, context_outputsize
        if not self.cpc_train_mode:
            return latents, context, hidden

        loss = torch.Tensor([0.0]).cuda()
        correct = 0 #will become Tensor
        for k in range(0, n_timesteps_out): #Do this for each timestep
            latent_k = latents[-n_timesteps_out + k] #batches, latents
            pred_k = self.predictor(context, k) # Shape (Batches, latents)
            softmax = self.lsoftmax(torch.mm(latent_k, pred_k.T)) #output: (Batches, Batches)
            correct += torch.sum(torch.argmax(softmax, dim=0) == torch.arange(n_batches).cuda())
            loss += torch.sum(torch.diag(softmax))
        loss /= (n_batches * n_timesteps_out) * -1.0
        accuracy = correct.true_divide(n_batches*n_timesteps_out)
        return accuracy, loss, hidden




    # Calculate the loss
    # future_latents = []
    # predicted_latents = []
    # for k in range(0, n_timesteps_out):  # Do this for each timestep
    #     future_latents.append(self.encoder(x_windows[-n_timesteps_out + k]))  # batches, latents
    #     predicted_latents.append(self.predictor(context, k))  # Shape (Batches, latents)
    # loss = self.info_NCE_loss_brian(torch.stack(future_latents), torch.stack(predicted_latents))
    #  #Will become Tensor
    # return torch.Tensor([0.0]).cuda(), loss, hidden
    def info_NCE_loss_brian(self, target_latents: torch.Tensor, pred_latents:torch.Tensor, emb_scale=0):
        """
        Calculates the infoNCE loss for CPC according to 'DATA-EFFICIENT IMAGE RECOGNITION
        WITH CONTRASTIVE PREDICTIVE CODING' A.2
        # latents: [B, H, W, D]
        loss = 0 . 0
        context = pixelCNN(latents)
        targets = Conv2D(output_channels=target_dim ,kernel_shape= ( 1 , 1 ) ) (latents)

        batch_dim , rows = targets . shape [ : - 1 ]
        targets = reshape(targets , [ - 1 , target_dim ] )
        for i in range(steps_to_ignore , steps_to_predict) :
            total_elements = batch_dim ∗ rows
            preds_i = Conv2D(output_channels=target_dim , kernel_shape= ( 1 , 1 ) ) (context)
            preds_i = preds_i [ : , : - (i+ 1 ) , : , : ] ∗ emb_scale
            preds_i = reshape(preds_i , [ - 1 , target_dim ] )
            logits = matmul(preds_i , targets , transp_b=True)
            b = range(total_elements) / (rows)
            col = range(total_elements) % (rows)
            labels = b ∗ rows + (i+1) ∗ rows + col
            loss += cross_entropy_with_logits(logits , labels)
        return loss
        """
        loss = torch.Tensor([0.0]).cuda()
        # latents = latents.permute(1, 0, 2)
        #batch_dim, col_dim, row_dim, _ = latents.shape
        timesteps_out, batch_dim, _ = target_latents.shape
        #targets = target_latents.transpose(1, 0)
        targets = target_latents.view([-1, self.latent_size])  # reshape
        #print(targets.shape)
        for i in range(timesteps_out):
            total_elements = batch_dim #* timesteps_out
            logits = torch.matmul(pred_latents[i], torch.t(targets[i])) #torch.t(targets)
            b = torch.arange(total_elements).cuda()
            col = b % timesteps_out
            labels = b + (i + 1) * timesteps_out + col
            #print(labels)
            temp_loss = torch.sum(- labels * F.log_softmax(logits, -1),
                                  -1)  # From https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
            mean_loss = temp_loss.mean()
            loss += mean_loss
        return loss


    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def set_train_mode(self, mode_bool):
        self.cpc_train_mode = mode_bool


    def test_modules(self, in_channels, window_size, latent_size, timesteps_in, timesteps_out):
        self._test_modules(1, in_channels, window_size, latent_size, timesteps_in, timesteps_out)
        for r in np.random.randint(2, 100, 10):
            self._test_modules(r, in_channels, window_size, latent_size, timesteps_in, timesteps_out)
        self._test_modules(128, in_channels, window_size, latent_size, timesteps_in, timesteps_out, verbose=True)

    def _test_modules(self, batch_size, in_channels, window_size, latent_size, timesteps_in, timesteps_out, verbose=False):
        enc_in = torch.rand((batch_size, in_channels, window_size))
        if verbose: print('Encoder input shape:', enc_in.shape)
        enc_out = self.encoder(enc_in)
        if verbose: print('Encoder output shape:', enc_out.shape)
        auto_in = torch.rand((timesteps_in)+enc_in.shape)
        if verbose: print('Autoregressive input shape (stacked encoder):', auto_in.shape)
        auto_out = self.autoregressive(auto_in)
        if verbose: print('Autoregressive output shape:', auto_out.shape)
        pred_in = auto_out[-1, :, :]
        if verbose: print('Predictor input shape:', pred_in.shape)
        pred_out = self.predictor(pred_in, timesteps_out)
        for i in range(1, timesteps_out):
            pred_out += self.predictor(pred_in, timesteps_out)
        if verbose: print('Predictor output shape:', pred_out.shape)
        assert enc_out.shape == pred_out.shape








