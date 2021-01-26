
import torch
from torch import nn
from torch import functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from cpc_utils import info_NCE_loss
from external.multi_scale_ori import MSResNet

class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        self.resnet = MSResNet(channels, layers=[1, 1, 1, 1], num_classes=2) #num_classes not really used

    def forward(self, x):
        # Input has shape (batches, channels, window_size)
        #print('Encoder input shape:', x.shape)
        #x = self.batch_norm(x)
        _, x = self.resnet(x)
        #x = x.permute(2, 0, 1).squeeze(0) #Only squeeze first (NOT BATCH!) dimension
        #print('Encoder output shape:', x.shape)

        return x