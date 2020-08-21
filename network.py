import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from params import *
import torchvision.models as models
from custom_resnet import customresnet
from collections import OrderedDict


class FeatureExtractor(nn.Module):
    def __init__(self, imheight, nc, n_layers, n_out, conv, batch_norm, max_pool, bool_resnet18=False,
                 bool_custom_resnet=False):
        """
        Feature extractor for the RCNN
        :param imheight: height of the images that will be given as input of the network.
        :param n_layers: number of convolutional layers of the feature extractor.
        :param n_out: number of channels that each convolutional layer should output.
        :param conv: dictionary that stocks info about the conv layers (kernel size, stride and padding).
        :param batch_norm: list of booleans : says if batch normalization is to be applied at the end of each conv layer
        :param max_pool: list : dictionary that stocks info about the maxpooling layers
        :param resnet: bool : use ResNet18 or not
        (kernel size, stride and padding).
        """
        super(FeatureExtractor, self).__init__()
        if bool_resnet18:
            resnet18network = models.resnet18(pretrained=False)
            resnet18network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            network = torch.nn.Sequential(*(list(resnet18network.children())[:-2]))

        elif bool_custom_resnet:
            custom_resnet_network = customresnet()
            # For a prediction size of 100 on an image of width 400
            # custom_resnet_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # For a prediction size of 200 on an image of width 400
            # custom_resnet_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2, 1), padding=3, bias=False)
            # to have smaller kernel
            custom_resnet_network.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False)
            network = torch.nn.Sequential(*(list(custom_resnet_network.children())[:-2]))
        else:
            network = nn.Sequential()
            # Create layers iteratively
            for k in range(n_layers):
                if k == 0:
                    n_in = nc
                else:
                    n_in = n_out[k-1]
                network.add_module('conv{0}'.format(k), nn.Conv2d(n_in,
                                                                  n_out[k],
                                                                  kernel_size=conv['kernel'][k],
                                                                  stride=conv['stride'][k],
                                                                  padding=conv['padding'][k]))
                if batch_norm[k]:
                    network.add_module('batchnorm{0}'.format(k), nn.BatchNorm2d(n_out[k]))
                network.add_module('relu{0}'.format(k), nn.ReLU())
                if max_pool['kernel'][k] != 0:
                    network.add_module('maxpool{0}'.format(k),
                                       nn.MaxPool2d(kernel_size=max_pool['kernel'][k],
                                                    stride=max_pool['stride'][k],
                                                    padding=max_pool['padding'][k]))

        # Create network
        self.network = network


class RNN(nn.Module):
    def __init__(self, n_layers, n_input, n_hidden, n_out, bidirectional=True, dropout=True):
        """
        The recurrent part of the RCNN
        :param n_layers: int: number of recurrent layers.
        :param n_input: int: number of input channels.
        :param n_hidden: int: number of hidden cells for each layer
        :param n_out: int :number of output channels
        :param bidirectional: boolean: If True, the layers created are LSTMs. If False, the layers are BLSTMs.
        """
        super(RNN, self).__init__()
        # Recurrent layers
        if dropout > 0:
            rnn = nn.Sequential(nn.LSTM(input_size=n_input,
                                        hidden_size=n_hidden,
                                        num_layers=n_layers,
                                        bidirectional=bidirectional,
                                        dropout=dropout))
        else:
            rnn = nn.Sequential(nn.LSTM(input_size=n_input,
                                        hidden_size=n_hidden,
                                        num_layers=n_layers,
                                        bidirectional=bidirectional))
        self.rnn = rnn

        # Linear layer
        if bidirectional:
            self.embedding = nn.Linear(n_hidden * 2, n_out)
        else:
            self.embedding = nn.Linear(n_hidden, n_out)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(t, b, -1)
        return output


class RCNN(nn.Module):
    """ RCNN for HTR """
    def __init__(self, imheight, nc, n_conv_layers, n_conv_out, conv, batch_norm,
                 max_pool, n_r_layers, n_hidden, n_out, bidirectional=True, resnet18=False, custom_resnet=False,
                 dropout=0.0):
        super(RCNN, self).__init__()
        self.featextractor = FeatureExtractor(imheight, nc, n_conv_layers, n_conv_out, conv, batch_norm, max_pool,
                                              resnet18, custom_resnet)
        self.recnet = RNN(n_r_layers, n_conv_out[-1], n_hidden, n_out, bidirectional, dropout)

    def forward(self, input):
        # Feature extractor
        conv = self.featextractor.network(input)
        # print('size after feature extractor:', conv.shape)

        # Convert output for RNN
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        # print("conv =", conv)
        # Recurrent network
        output = self.recnet(conv)
        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)
        return output


if __name__ == "__main__":
    params, log_dir = BaseOptions().parser()

    print('Example of usage')
    x = torch.randn(8, 1, 64, 800)  # nSamples, nChannels, Height, Width
    print('x', x.shape)

    fullrcnn = RCNN(imheight=params.imgH,
                    nc=params.NC,
                    n_conv_layers=params.N_CONV_LAYERS,
                    n_conv_out=params.N_CONV_OUT,
                    conv=params.CONV,
                    batch_norm=params.BATCH_NORM,
                    max_pool=params.MAX_POOL,
                    n_r_layers=params.N_REC_LAYERS,
                    n_hidden=params.N_HIDDEN,
                    n_out=params.N_CHARACTERS,
                    bidirectional=True, resnet18=False, custom_resnet=True)
    # The arguments of RCNN are defined in params.py
    # print('Network \n', fullrcnn)

    zbis = fullrcnn(x)
    print('zbis', zbis.shape)

    #f = fullrcnn.featextractor.network._modules['conv0'](x)
    #print(f.shape)

