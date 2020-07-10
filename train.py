import torch as torch
import os
import network
import params
import data_utils
from torch.nn import CTCLoss
import torch.nn as nn
from myDataset import myDataset
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init():
    nclass = len(params.alphabet)
    rcnn = network.RCNN(imheight=params.imgH,
                        nc=params.NC,
                        n_conv_layers=params.N_CONV_LAYERS,
                        n_conv_out=params.N_CONV_OUT,
                        conv=params.CONV,
                        batch_norm=params.BATCH_NORM,
                        max_pool=params.MAX_POOL,
                        n_r_layers=params.N_REC_LAYERS,
                        n_hidden=params.N_HIDDEN,
                        n_out=params.N_CHARACTERS,
                        bidirectional=params.BIDIRECTIONAL)

    if params.pretrained != '':
        print('Loading pretrained model from %s' % params.pretrained)
        # if params.multi_gpu:
        #    rcnn = torch.nn.DataParallel(rcnn)
        rcnn.load_state_dict(torch.load(params.pretrained))
        print('Loading done.')
    elif params.weights_init:
        rcnn.apply(weights_init)

    return rcnn

# -----------------------------------------------
"""
In this block
    training function 
    evaluation function -> To do 
"""


def train(model, criterion, optimizer, train_loader):
    print("Starting training...")
    losses = []
    # Set requires_grad to True & set model mode to train & initialize optimizer gradients
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    optimizer.zero_grad()

    for epoch in range(10): #params.epochs
        avg_cost = 0
        for iter_idx, (img, transcr) in enumerate(tqdm(train_loader)):
            # Process predictions
            img = Variable(img.data.unsqueeze(1))
            if params.cuda and torch.cuda.is_available():
                img = img.cuda()
            # print("img =", img)
            preds = model(img)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * img.size(0)))
            # print("preds_size =", preds_size)
            # Process labels
            # CTCLoss().cuda() only works with LongTensor
            labels = Variable(torch.LongTensor([params.cdict[c] for c in ''.join(transcr)]))
            label_lengths = torch.LongTensor([len(t) for t in transcr])
            # criterion = CTC loss
            if params.cuda and torch.cuda.is_available():
                preds_size = preds_size.cuda()
                labels = labels.cuda()
                label_lengths = label_lengths.cuda()
            cost = criterion(preds, labels, preds_size, label_lengths)# / batch_size
            avg_cost += cost
            # print("avg_cost = ", avg_cost)
            cost.backward()
            optimizer.step()

        avg_cost = avg_cost/len(train_loader)
        # print('avg_cost', avg_cost.item())
        losses.append(avg_cost.item())
        #print("img = ", img)
        #print("preds = ", preds)
        #print("labels = ", labels)
        print("preds_size = ", preds_size)
        print("label_lengths = ", label_lengths)
        print('Epoch[%d/%d] Average Loss: %f' % (epoch+1, params.epochs, avg_cost))

    #plt.plot(np.arange(params.epochs), losses)
    #plt.title("Average losses during training")
    #plt.show()
    print("Average losses during training", losses)
    print("Training done.")
    return losses



# -----------------------------------------------
"""
In this block
    initialise the model
"""
MODEL = net_init()
#print(MODEL)
if params.cuda and torch.cuda.is_available():
    # MODEL = nn.DataParallel(MODEL)
    MODEL = MODEL.cuda()
# -----------------------------------------------
"""
In this block
    criterion define
"""
CRITERION = CTCLoss()
if params.cuda and torch.cuda.is_available():
    CRITERION = CRITERION.cuda()

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
else:
    OPTIMIZER = optim.RMSprop(MODEL.parameters(), lr=params.lr)

# -----------------------------------------------

if __name__ == "__main__":
    # when data_size = (32, None), the width is not fixed
    train_set = myDataset(data_size=(32, None), set='train')
    test_set = myDataset(data_size=(32, None), set='test')
    val1_set = myDataset(data_size=(32, None), set='val1')
    print("len(train_set) =", train_set.__len__())
    print("len(test_set) =", test_set.__len__())
    print("len(val1_set) =", val1_set.__len__())
    # augmentation using data sampler
    batch_size = 5
    TRAIN_LOADER = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                              collate_fn=data_utils.pad_packed_collate)
    # train model
    train(MODEL, CRITERION, OPTIMIZER, TRAIN_LOADER)

    # eventually save model
    if params.save:
        torch.save(MODEL.state_dict(), '{0}/netRCNN.pth'.format(params.save_location))
        print("Network saved at location %s" % params.save_location)
