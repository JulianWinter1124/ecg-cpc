from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(n_channels[0]) #not used in paper?
        self.convolutionals = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=latent_size, kernel_size=3, dilation=1),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, dilation=2),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, dilation=4),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, dilation=8),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
        )

        def _weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x):
        # Input has shape (batches, channels, window_size)
        # print('Encoder input shape:', x.shape)
        # x = self.batch_norm(x)
        x = self.convolutionals(x)
        # print('Encoder output shape:', x.shape)

        return x
