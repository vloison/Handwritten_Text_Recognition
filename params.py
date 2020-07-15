
alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """

cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet)} # int -> character
# '_' is the blank character for CTC


import os
import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False
        root_path = '/media/vn_nguyen/hdd/hux/Results/'
        self.log_dir = root_path + 'test{}'.format(len(os.listdir(root_path)) + 1)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def initialize(self, parser):
        parser.add_argument('--log_dir', type=str, default=self.log_dir)
        # DATA PARAMETERS
        parser.add_argument('--imgH', type=int, default=32)
        parser.add_argument('--imgW', type=int, default=400)
        # PARAMETERS FOR LOADING/SAVING NETWORKS
        parser.add_argument('--weights_init', type=bool, default=True)
        parser.add_argument('--pretrained', type=str, default='')  #
        parser.add_argument('--save', type=bool, default=True, help='Whether to save the trained network')
        # TRAINING PARAMETERS
        parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA or not')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--epochs', type=int, default=20, help='Training epoch number')
        # Optimizer
        parser.add_argument('--adam', type=bool, default=False, help='Use Adam or not')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
        parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate')
        # PARAMETERS FOR THE FEATURE EXTRACTOR
        parser.add_argument('--N_CONV_LAYERS', type=int, default=7)
        parser.add_argument('--NC', type=int, default=1, help='Number of channels given as an input of RCNN')
        # Convolutional layers
        parser.add_argument('--N_CONV_OUT', type=list, default=[64, 128, 256, 256, 512, 512, 512])
        parser.add_argument('--CONV', type=dict, default={
            'kernel': [3, 3, 3, 3, 3, 3, 2],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'padding': [1, 1, 1, 1, 1, 1, 0]
        })
        # Batch normalization
        parser.add_argument('--BATCH_NORM', type=list, default=[False, False, True, False, True, False, True])
        # Maxpooling
        parser.add_argument('--MAX_POOL', type=dict, default={
            'kernel': [2, 2, 0, (2, 2), 0, (2, 2), 0],
            'stride': [2, 2, 0, (2, 1), 0, (2, 1), 0],
            'padding': [0, 0, 0, (0, 1), 0, (0, 1), 0]
        })
        # PARAMETERS FOR THE RECURRENT NETWORK
        parser.add_argument('--N_REC_LAYERS', type=int, default=2)
        parser.add_argument('--N_HIDDEN', type=int, default=256)
        parser.add_argument('--N_CHARACTERS', type=int, default=len(alphabet))
        parser.add_argument('--BIDIRECTIONAL', type=bool, default=True, help='Use bidirectional LSTM or not')

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '---------------------Options------------------\n'
        for k,v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v),comment)
        message += '----------------------End---------------------\n'
        print(message)

        opt_file = os.path.join(self.log_dir,'params.txt')
        with open(opt_file,'w') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parser(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        self.opt = opt
        self.print_options(self.opt)
        return self.opt, self.log_dir
