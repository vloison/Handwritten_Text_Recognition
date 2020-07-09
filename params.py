alphabet = """"_!"#&\()*+,-.'/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
# '_' is the blank character for CTC

# PARAMETERS FOR THE FEATURE EXTRACTOR
N_CONV_LAYERS = 7
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
MP_PADDINGS = [0, 0, 0, (1, 1), 0, (1, 1), 0]
MAX_POOL = {
    'kernel': MP_KERNEL_SIZES,
    'stride': MP_STRIDES,
    'padding': MP_PADDINGS
}
# PARAMETERS FOR THE RECURRENT NETWORK
N_REC_LAYERS = 2
N_HIDDEN = 256
N_CHARACTERS = 26