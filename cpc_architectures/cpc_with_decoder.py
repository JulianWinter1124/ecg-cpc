import math

from torch import nn
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import ecg_datasets2
from cpc_architectures import cpc_encoder_v1, cpc_encoder_v2, cpc_autoregressive_v0, cpc_encoder_decoder_v2
from external import tcn
from external.tcn.TCN.tcn import TemporalConvNet


class SimpleCPC(nn.Module):

    def __init__(self, encoder, decoder, autoregressive, timesteps_in, timesteps_out, latent_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.autoregressive = autoregressive
        self.timesteps_in = timesteps_in
        self.timesteps_out = timesteps_out
        self.latent_size = latent_size

    def forward(self, X):
        batch, n_windows, channels, length = X.shape
        assert n_windows >= self.timesteps_in + self.timesteps_out
        x = X.reshape((batch*n_windows, channels, length)) #squash into batch dimension
        encoded_x = self.encoder(x)
        encoded_x = encoded_x.reshape((batch, n_windows, -1)) #reshape into original
        # if self.autoregressive.uses_hidden_state():
        #     if hidden is None:
        #         hidden = self.autoregressive.init_hidden(batch, use_gpu=False)
        #     predicted, hidden = self.autoregressive(encoded_x[:self.timesteps_in], hidden)
        # else:
        encoded_x = encoded_x.transpose(1, 2) #new shape is (batch, latents, steps)
        predicted = self.autoregressive(encoded_x[:, :, :self.timesteps_in])
        predicted_timesteps_out = predicted[:, :, -self.timesteps_out:].transpose(1, 2).reshape((batch*self.timesteps_out, -1))
        #encoded_x_timesteps_out = encoded_x[:, :, -self.timesteps_out:].transpose(1, 2).reshape((batch*self.timesteps_out, -1))
        decoded_pred = self.decoder(predicted_timesteps_out).reshape((batch, self.timesteps_out, channels, length))
        if self.training:
            loss = torch.sum(torch.square(decoded_pred-X[:, -self.timesteps_out:]))/(batch*self.timesteps_out)
            return loss
        else:
            return decoded_pred





if __name__ == '__main__':
    timesteps_in = 6
    timesteps_out = 6
    latents = 64
    enc = cpc_encoder_v2.Encoder(12, latents)
    auto = TemporalConvNet(latents, [latents]*3, kernel_size=3)
    decoder = cpc_encoder_decoder_v2.Decoder(12, latents)
    scpc = SimpleCPC(enc, decoder, auto, timesteps_in, timesteps_out, latents)
    train = ecg_datasets2.ECGChallengeDatasetBatching('/media/julian/data/data/ECG/ptbxl_challenge',
                                                           window_size=140, n_windows=timesteps_in+timesteps_out)
    val = ecg_datasets2.ECGChallengeDatasetBatching('/media/julian/data/data/ECG/ptbxl_challenge',
                                                           window_size=140, n_windows=timesteps_in+timesteps_out)
    dataloader = DataLoader(train, batch_size=128, drop_last=True, num_workers=1)
    valloader = DataLoader(val, batch_size=128, drop_last=True, num_workers=1)
    optimizer = Adam(scpc.parameters())
    scpc.train()
    for i in range(2000):
        avg_train_loss, avg_test_loss = 0, 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            loss = scpc(data.float())
            loss.backward()
            optimizer.step()
            avg_train_loss = (avg_train_loss*batch_idx+loss.item())/(batch_idx+1)
            print("loss {}".format(loss.item()))
        print('EPOCH {} finished. Average train loss {}'.format(i+1, avg_train_loss))
        # for batch_idx, data in enumerate(dataloader):
        #     loss = scpc(data.float())
        #     avg_test_loss = (avg_test_loss*batch_idx+loss.item())/(batch_idx+1)
        # print('EPOCH {} finished. Average validation loss {}'.format(i+1, avg_test_loss))
    scpc.train(False)
