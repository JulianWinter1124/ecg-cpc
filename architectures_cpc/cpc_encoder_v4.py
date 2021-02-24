from torch import nn
#Idea: Make a bug receptive field using dilations
#Do not use strides to not miss any information
#Use maxpool instead of strides to select a window where activation is biggest

class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        kernel_sizes = [5, 3, 3, 3, 3]
        strides = [1, 1, 1, 1, 1]
        dilations = [1, 5, 3*5, 3*3*5, 5*3*3*3]
        max_pool_sizes = [3, 3, 3, 3, 1]
        n_channels = [channels] + [latent_size] * len(kernel_sizes)
        #self.batch_norm = nn.BatchNorm1d(n_channels[0]) #not used in paper?
        self.convolutionals = nn.Sequential(
            *[e for t in [
                (nn.Conv1d(in_channels=n_channels[i], out_channels=n_channels[i + 1], kernel_size=kernel_sizes[i], dilation=dilations[i], stride=strides[i], padding=0),
                 nn.MaxPool1d(max_pool_sizes[i]),
                 (nn.ReLU())
                 ) for i in range(len(kernel_sizes))] for e in t]
        )

        #self.avg_pool = nn.AdaptiveAvgPool1d(1)
        def _weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x):
        # Input has shape (batches, channels, window_size)
        #print('Encoder input shape:', x.shape)
        #x = self.batch_norm(x)

        x = self.convolutionals(x)
        #print('out shape', x.shape)
        #x = self.avg_pool(x)
        #print('out2', x.shape)
        # Output has shape (batches, latents, 1)
        #Maybe squeeze??
        x = x.squeeze(2) #Only squeeze first (NOT BATCH!) dimension
        #print('Encoder output shape:', x.shape)

        return x
