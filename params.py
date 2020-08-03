import os
import argparse
import time

# alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
alphabet = [' ', '!', '"', '&', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
              'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~', '°',
              'é', '§', '$', '+', '%', "'", '©', '|', '\\', '#', '@', '£', '€', '®']

cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet)}  # int -> character
# '_' is the blank character for CTC


class BaseOptions():
    def __init__(self):
        self.initialized = False
        root_path = '/media/vn_nguyen/hdd/hux/Results/'
        self.log_dir = root_path + time.strftime("%m-%d_%H:%M:%S", time.localtime())
        # self.log_dir = root_path + '07-31_16:15:27'
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def initialize(self, parser):
        parser.add_argument('--log_dir', type=str, default=self.log_dir)
        # DATA PARAMETERS
        parser.add_argument('--imgH', type=int, default=32)
        parser.add_argument('--imgW', type=int, default=3200)
        parser.add_argument('--data_aug', type=bool, default=False)
        # PARAMETERS FOR LOADING/SAVING NETWORKS
        parser.add_argument('--train', type=bool, default=True, help='Train a network or not')
        parser.add_argument('--weights_init', type=bool, default=True)
        parser.add_argument('--pretrained', type=str, default='')  #
        # parser.add_argument('--pretrained', type=str,
        #                     default='/media/vn_nguyen/hdd/hux/Results_network/SGD/07-31_16:15:27/netRCNN.pth')
        parser.add_argument('--save', type=bool, default=True, help='Whether to save the trained network')
        # PARAMETERS FOR PLOT
        parser.add_argument('--previous_epochs', type=int, default=0)
        # TRAINING PARAMETERS
        parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA or not')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--epochs', type=int, default=400, help='Training epoch number')
        # Optimizer
        parser.add_argument('--milestones', type=list, default=[150], help='Milestones to change lr')
        parser.add_argument('--adam', type=bool, default=False, help='Use Adam or not')
        parser.add_argument('--adadelta', type=bool, default=False, help='Use ADADELTA or not')
        parser.add_argument('--sgd', type=bool, default=False, help='Use SGD or not')
        parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
        parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
        parser.add_argument('--rho', type=float, default=0.9, help='rho for ADADELTA')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 penalty) ')
        # PARAMETERS FOR THE FEATURE EXTRACTOR
        parser.add_argument('--RESNET18', type=bool, default=True)  # if using resnet18, we need imgW at least 3200
        parser.add_argument('--N_CONV_LAYERS', type=int, default=7)  # 7
        parser.add_argument('--NC', type=int, default=1, help='Number of channels given as an input of RCNN')
        # Convolutional layers
        parser.add_argument('--N_CONV_OUT', type=list,
                            default=[64, 128, 256, 256, 512, 512, 512]  # [16, 32, 64, 128] #
                            )
        parser.add_argument('--CONV', type=dict, default={
            'kernel': [3, 3, 3, 3, 3, 3, 2],  # [3,3,3,3], #
            'stride': [1, 1, 1, 1, 1, 1, 1],  # [1,1,1,1], #
            'padding': [1, 1, 1, 1, 1, 1, 0]  # [1,1,1,1] #
        })
        # Batch normalization
        parser.add_argument('--BATCH_NORM', type=list,
                            default=[False, False, True, False, True, False, True]  # [True, True, True, True] #
                            )
        # Maxpooling
        parser.add_argument('--MAX_POOL', type=dict, default={
            'kernel': [2, 2, 0, (2, 2), 0, (2, 2), 0],  # [2,2,2,4], #
            'stride': [2, 2, 0, (2, 1), 0, (2, 1), 0],  # [2,2,2,4], #
            'padding': [0, 0, 0, (0, 1), 0, (0, 1), 0]  # [0,0,0,0] #
        })
        # PARAMETERS FOR THE RECURRENT NETWORK
        parser.add_argument('--N_REC_LAYERS', type=int, default=1)  # 2
        parser.add_argument('--N_HIDDEN', type=int, default=256)
        parser.add_argument('--N_CHARACTERS', type=int, default=len(alphabet))
        parser.add_argument('--BIDIRECTIONAL', type=bool, default=True, help='Use bidirectional LSTM or not')
        parser.add_argument('--DROPOUT', type=float, default=0.0, help='Dropout parameter within [0,1] in BLSTM')

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '---------------------Options------------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------------End---------------------\n'
        print(message)

        opt_file = os.path.join(self.log_dir, 'params.txt')
        with open(opt_file, 'w') as opt_file:
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
