
alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """

cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet)} # int -> character
# '_' is the blank character for CTC

# DATA PARAMETERS
imgH = 32
imgW = 400

# PARAMETERS FOR LOADING/SAVING NETWORKS
weights_init = True
pretrained = 'trained_networks/netRCNN.pth'  # 'trained_networks/netRCNN.pth'
save = False  # Whether to save the trained network
save_location = 'trained_networks'
# Path to the pretrained model to continue training. if pretrained == '', a new network will be created and trained.

# TRAINING PARAMETERS
cuda = True
# Optimizer
adam = True  # I only put adam for now, but in the github there is also ADADELTA and RMSprop
lr = 0.0001  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
epochs = 2  # training epoch number
displayInterval = 100
# I copy-pasted the values from the HolmesYoung github, maybe try to change the values later

# PARAMETERS FOR THE FEATURE EXTRACTOR
N_CONV_LAYERS = 7
NC = 1  # Number of channels given as an input of RCNN : =1 because 2D images are given as arguments

# Convolutional layers
N_CONV_OUT = [64, 128, 256, 256, 512, 512, 512]
CONV_KERNEL_SIZES = [3, 3, 3, 3, 3, 3, 2]
CONV_STRIDES = [1, 1, 1, 1, 1, 1, 1]
CONV_PADDINGS = [1, 1, 1, 1, 1, 1, 0]
CONV = {
    'kernel': CONV_KERNEL_SIZES,
    'stride': CONV_STRIDES,
    'padding': CONV_PADDINGS
}
# Batch normalization
BATCH_NORM = [False, False, True, False, True, False, True]
# Maxpooling
MP_KERNEL_SIZES = [2, 2, 0, (2, 2), 0, (2, 2), 0]
MP_STRIDES = [2, 2, 0, (2, 1), 0, (2, 1), 0]
MP_PADDINGS = [0, 0, 0, (0, 1), 0, (0, 1), 0]
MAX_POOL = {
    'kernel': MP_KERNEL_SIZES,
    'stride': MP_STRIDES,
    'padding': MP_PADDINGS
}
# PARAMETERS FOR THE RECURRENT NETWORK
N_REC_LAYERS = 2
N_HIDDEN = 256
N_CHARACTERS = len(alphabet)
print(N_CHARACTERS)
BIDIRECTIONAL = True
