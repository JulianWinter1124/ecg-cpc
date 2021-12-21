from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        filters = [11, 7, 5, 5, 3]   # See https://arxiv.org/pdf/1807.03748.pdf#Audio
        strides=[3, 2, 2, 1, 1]
        n_channels = [channels] + [latent_size] * len(filters)
        # self.batch_norm = nn.BatchNorm1d(n_channels[0]) #not used in paper?
        self.convolutionals = nn.Sequential(
            *[e for t in [
                (nn.Conv1d(in_channels=n_channels[i], out_channels=n_channels[i + 1], kernel_size=filters[i],
                           stride=strides[i], padding=0),
                 (nn.BatchNorm1d(n_channels[i + 1])),
                 (nn.ReLU())
                 ) for i in range(len(filters))] for e in t]
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
