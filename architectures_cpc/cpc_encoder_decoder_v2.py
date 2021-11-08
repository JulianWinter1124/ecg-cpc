from torch import nn


class Decoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Decoder, self).__init__()
        filters = [8, 6, 3, 3, 3]  # , (8,1), (4,1), (4,1), (4,1)]  # See https://arxiv.org/pdf/1807.03748.pdf#Audio
        strides = [4, 2, 1, 1, 1]  # , (4,1), (2,1), (2,1), (1,1)]
        dilations = [1, 1, 1, 2, 4]
        n_channels = [channels] + [latent_size] * len(filters)
        # self.batch_norm = nn.BatchNorm1d(n_channels[0]) #not used in paper?
        self.deconvolutionals = nn.Sequential(
            *[e for t in [
                (nn.ConvTranspose1d(in_channels=n_channels[i + 1], out_channels=n_channels[i], kernel_size=filters[i],
                                    dilation=dilations[i], stride=strides[i], padding=0),
                 (nn.BatchNorm1d(n_channels[i])),
                 (nn.ReLU())
                 ) for i in reversed(range(len(filters)))] for e in t]
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        def _weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x):
        # Input has shape (batches, channels, window_size)
        # print('decoder input shape:', x.shape)
        # x = self.batch_norm(x)
        x = self.deconvolutionals(x.unsqueeze(2))
        x = x.squeeze(2)  # Only squeeze first (NOT BATCH!) dimension
        # print('Encoder output shape:', x.shape)

        return x
