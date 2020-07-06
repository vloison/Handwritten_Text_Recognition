import torch.nn as nn
from collections import OrderedDict

N_LAYERS = 3
N_OUT = (20, 30, 10)
KERNELS = (5, 6, (7, 6))
STRIDES = (1, 1, 1)
BATCH_NORM = [False, True, False]
MAX_POOL = [(2, 2), 0, 0]


class FeatureExtractor(nn.Module):
    def __init__(self, imheight, n_layers, n_out, kernels, strides, batch_norm, max_pool):
        """imheight :
        :param imheight: height of the images that will be given as input of the network.
        The height needs to be fixed, not the width.
        :param n_layers: number of layers of the feature extractor
        :param n_out: number of channels that each convolutional layer should output.
        :param kernels: list : size of the kernel of each convolutional layer
        :param strides: list : stride for each convolutional layer
        :param batch_norm: list of booleans : says if batch normalization is to be applied at the end of each layer
        :param max_pool: list : says if maxpooling is to bo applied at the end of each layer, and the size of the
        eventual maxpooling
        """
        super(FeatureExtractor, self).__init__()
        list_layers = []
        # Create layers iteratively
        for k in range(n_layers):
            if k == 0:
                n_in = imheight
            else:
                n_in = n_out[k-1]
            list_layers.append(('conv{0}'.format(k), nn.Conv2d(n_in, n_out[k], kernel_size=kernels[k], stride=strides[k])))
            if batch_norm[k]:
                list_layers.append(('batchnorm{0}'.format(k), nn.BatchNorm2d(n_out[k])))
            list_layers.append(('relu{0}'.format(k), nn.ReLU()))
            if max_pool[k] != 0:
                list_layers.append(('maxpool{0}'.format(k), nn.MaxPool2d(kernel_size=max_pool[k])))

        # Create network
        self.network = nn.Sequential(OrderedDict(list_layers))


model = FeatureExtractor(imheight=16, n_layers=N_LAYERS, n_out=N_OUT, kernels=KERNELS, strides=STRIDES, batch_norm=BATCH_NORM, max_pool=MAX_POOL)
print(model.network)
