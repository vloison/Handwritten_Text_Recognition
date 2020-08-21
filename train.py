import torch as torch
import os
import shutil
import network
from params import *
import data.IAM_dataset
from torch.nn import CTCLoss
import data.Preprocessing
from data.myDataset import myDataset
from data.myDataset import lmdbDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import CER, WER

# ------------------------------------------------
"""
In this block
    Set path to log
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

params, log_dir = BaseOptions().parser()
print("log_dir =", log_dir)
if params.save:
    writer = SummaryWriter(log_dir)  # TensorBoard(log_dir)
else:
    shutil.rmtree(log_dir)

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
                        bidirectional=params.BIDIRECTIONAL,
                        resnet18=params.RESNET18,
                        custom_resnet=params.custom_resnet,
                        dropout=params.DROPOUT)

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
    evaluation function
"""


def test(model, criterion, test_loader, len_test_set):
    print("Starting testing...")
    model.eval()

    avg_cost = 0
    avg_CER = 0
    avg_WER = 0
    for iter_idx, (img, transcr) in enumerate(tqdm(test_loader)):
        # Process predictions
        img = Variable(img.data.unsqueeze(1))
        if params.cuda and torch.cuda.is_available():
            img = img.cuda()
        # print(img.type)
        with torch.no_grad():
            preds = model(img)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * img.size(0)))

        # Process labels for CTCLoss
        labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(transcr)]))
        label_lengths = torch.LongTensor([len(t) for t in transcr])
        # Compute CTCLoss
        if params.cuda and torch.cuda.is_available():
            preds_size = preds_size.cuda()
            labels = labels.cuda()
            label_lengths = label_lengths.cuda()
        cost = criterion(preds, labels, preds_size, label_lengths)  # / batch_size
        avg_cost += cost.item()

        # Convert paths to string for metrics
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()

        if tdec.ndim == 1:  # If the batch has size 1
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            # Compute metrics
            avg_CER += CER(transcr[0], dec_transcr)
            avg_WER += WER(transcr[0], dec_transcr)
        else:
            for k in range(len(tdec)):
                tt = [v for j, v in enumerate(tdec[k]) if j == 0 or v != tdec[k][j - 1]]
                dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
                # Compute metrics
                avg_CER += CER(transcr[k], dec_transcr)
                avg_WER += WER(transcr[k], dec_transcr)

                if iter_idx % 50 == 0 and k % 2 == 0:
                    print('label:', transcr[k])
                    print('prediction:', dec_transcr)
                    print('CER:', CER(transcr[k], dec_transcr))
                    print('WER:', WER(transcr[k], dec_transcr))
                    if params.save:
                        writer.add_text(transcr[k],
                                        dec_transcr + '  --[CER=' + str(
                                            round(CER(transcr[k], dec_transcr), 2)) + ']', 0)
                        writer.add_text(transcr[k],
                                        dec_transcr + '  --[WER=' + str(
                                            round(WER(transcr[k], dec_transcr), 2)) + ']', 0)

    avg_cost = avg_cost / len(test_loader)
    avg_CER = avg_CER / len_test_set
    avg_WER = avg_WER / len_test_set
    print('Average CTCloss', avg_cost)
    print("Average CER", avg_CER)
    print("Average WER", avg_WER)

    print("Testing done.")
    return avg_cost, avg_CER, avg_WER


def val(model, criterion, val_loader, len_val_set):
    model.eval()
    avg_cost = 0
    avg_CER = 0
    avg_WER = 0

    for iter_idx, (img, transcr) in enumerate(tqdm(val_loader)):
        # Process predictions
        img = Variable(img.data.unsqueeze(1))
        if params.cuda and torch.cuda.is_available():
            img = img.cuda()
        # print(img.type)
        with torch.no_grad():
            preds = model(img)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * img.size(0)))

        # Process labels for CTCLoss
        labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(transcr)]))
        label_lengths = torch.LongTensor([len(t) for t in transcr])
        # Compute CTCLoss
        if params.cuda and torch.cuda.is_available():
            preds_size = preds_size.cuda()
            labels = labels.cuda()
            label_lengths = label_lengths.cuda()
        cost = criterion(preds, labels, preds_size, label_lengths)  # / batch_size
        avg_cost += cost.item()

        # Convert paths to string for metrics
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        if tdec.ndim == 1:
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            # Compute metrics
            avg_CER += CER(transcr[0], dec_transcr)
            avg_WER += WER(transcr[0], dec_transcr)
        else:
            for k in range(len(tdec)):
                tt = [v for j, v in enumerate(tdec[k]) if j == 0 or v != tdec[k][j - 1]]
                dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
                # Compute metrics
                avg_CER += CER(transcr[k], dec_transcr)
                avg_WER += WER(transcr[k], dec_transcr)

    avg_cost = avg_cost / len(val_loader)
    avg_CER = avg_CER / len_val_set
    avg_WER = avg_WER / len_val_set
    return avg_cost, avg_CER, avg_WER


def train(model, criterion, optimizer, lr_scheduler, train_loader, val_loader, len_val_set):

    print("Starting training...")
    losses = []
    print("optimizer.param_groups[0]['lr'] at beginning of training", optimizer.param_groups[0]['lr'])
    optimizer.zero_grad()

    for epoch in range(params.epochs):
        # Training
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        avg_cost = 0
        for iter_idx, (img, transcr) in enumerate(tqdm(train_loader)):
            # Process predictions
            # print("img =", img[0])
            img = Variable(img.data.unsqueeze(1))
            if params.cuda and torch.cuda.is_available():
                img = img.cuda()
            preds = model(img)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * img.size(0)))
            # Process labels
            # CTCLoss().cuda() only works with LongTensor
            labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(transcr)]))
            label_lengths = torch.LongTensor([len(t) for t in transcr])
            # criterion = CTC loss
            if params.cuda and torch.cuda.is_available():
                preds_size = preds_size.cuda()
                labels = labels.cuda()
                label_lengths = label_lengths.cuda()
            cost = criterion(preds, labels, preds_size, label_lengths)  # / batch_size

            avg_cost += cost.item()
            cost.backward()
            optimizer.step()
            lr_scheduler.step()
            # del preds_size, labels, label_lengths, cost
            # del img, preds, preds_size, labels, label_lengths, cost
            # if iter_idx > 0 and iter_idx % 100 == 0:
            #     print('Epoch[%d/%d] Avg Training Loss: %f'
            #           % (epoch + 1, params.epochs, avg_cost/(iter_idx*params.batch_size)))


        avg_cost = avg_cost/len(train_loader)

        # # log the loss
        if params.save:
            writer.add_scalar('train loss', avg_cost, params.previous_epochs + epoch)
        # Convert paths to string for metrics
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        if tdec.ndim == 1:  # If the batch has size 1
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        else:
            tt = [v for j, v in enumerate(tdec[0]) if j == 0 or v != tdec[0][j - 1]]

        print("sample img\n", img.shape)
        print("sample gt\n", transcr[0])
        print("lables\n", labels)
        print("lable_len = ", label_lengths)
        if params.save:
            dec_transcr = 'Train epoch ' + str(epoch).zfill(4) + ' Prediction ' + ''.join(
                [icdict[t] for t in tt]).replace('_', '')
            writer.add_image(dec_transcr, img[0], params.previous_epochs + epoch)
            # Save model
            torch.save(model.state_dict(), '{0}/netRCNN.pth'.format(log_dir))

        # Validation
        if epoch % 5 == 0:
            val_loss, val_CER, val_WER = val(model, criterion, val_loader, len_val_set)
            if params.save:
                writer.add_scalar('val loss', val_loss, params.previous_epochs + epoch)
                writer.add_scalar('val CER', val_CER, params.previous_epochs + epoch)
                writer.add_scalar('val WER', val_WER, params.previous_epochs + epoch)

        losses.append(avg_cost)
        # print("img = ", img.shape)
        # print("preds = ", preds)
        # print("labels = ", labels)
        # print("preds_size = ", preds_size)
        # print("label_lengths = ", label_lengths)

        print('Epoch[%d/%d] lr = %f \n Avg Training Loss: %f  Avg Validation loss: %f \n Avg CER: %f  Avg WER: %f'
              % (epoch+1, params.epochs, optimizer.param_groups[0]['lr'], avg_cost, val_loss, val_CER, val_WER))

    print("Training done.")
    return losses


# -----------------------------------------------
"""
In this block
    criterion define
"""
CRITERION = CTCLoss()
if params.cuda and torch.cuda.is_available():
    CRITERION = CRITERION.cuda()

# -----------------------------------------------

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Initialize model
    MODEL = net_init()
    # print(MODEL)
    if params.cuda and torch.cuda.is_available():
        MODEL = MODEL.cuda()

    # Initialize optimizer
    if params.optim_state == '':
        if params.adam:
            OPTIMIZER = optim.Adam(MODEL.parameters(), lr=params.lr, betas=(params.beta1, params.beta2),
                                   weight_decay=params.weight_decay)
        elif params.adadelta:
            OPTIMIZER = optim.Adadelta(MODEL.parameters(), lr=params.lr, rho=params.rho,
                                       weight_decay=params.weight_decay)
        elif params.sgd:
            OPTIMIZER = optim.SGD(MODEL.parameters(), lr=params.lr, momentum=params.momentum)
        else:
            OPTIMIZER = optim.RMSprop(MODEL.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    # Load optimizer state
    else:
        if params.adam:
            OPTIMIZER = optim.Adam(MODEL.parameters(), betas=(params.beta1, params.beta2),
                                   weight_decay=params.weight_decay)
        elif params.adadelta:
            OPTIMIZER = optim.Adadelta(MODEL.parameters(), rho=params.rho, weight_decay=params.weight_decay)
        elif params.sgd:
            OPTIMIZER = optim.SGD(MODEL.parameters(), momentum=params.momentum)
        else:
            OPTIMIZER = optim.RMSprop(MODEL.parameters(), weight_decay=params.weight_decay)
        print('Loading optimizer state from %s' % params.optim_state)
        OPTIMIZER.load_state_dict(torch.load(params.optim_state))
        print("Loading done.")

    # Load data
    # when data_size = (32, None), the width is not fixed
    train_set = myDataset(data_type=params.dataset, data_size=(params.imgH, params.imgW),
                          set='train', centered=False, deslant=False, data_aug=params.data_aug,  keep_ratio=params.keep_ratio,)
    test_set = myDataset(data_type=params.dataset, data_size=(params.imgH, params.imgW),
                         set='test', centered=False, deslant=False,  keep_ratio=params.keep_ratio)
    val1_set = myDataset(data_type=params.dataset, data_size=(params.imgH, params.imgW),
                         set='val', centered=False, deslant=False, keep_ratio=params.keep_ratio)

    # load OCR dataset
    # train_set = lmdbDataset(data_size=(params.imgH, params.imgW), dataset='train.easy')
    # test_set = lmdbDataset(data_size=(params.imgH, params.imgW), dataset='test.easy')
    # val1_set = lmdbDataset(data_size=(params.imgH, params.imgW), dataset='valid.easy')

    LEN_TRAIN_SET = train_set.__len__()
    LEN_TEST_SET = test_set.__len__()
    LEN_VAL1_SET = val1_set.__len__()
    print("len(train_set) =", LEN_TRAIN_SET)
    print("len(test_set) =", LEN_TEST_SET)
    print("len(val1_set) =", LEN_VAL1_SET)

    # print("optimizer.param_groups[0]['lr'] before LR_SCHEDULER", OPTIMIZER.param_groups[0]['lr'])
    # lr changing while training
    LR_SCHEDULER = MultiStepLR(OPTIMIZER,
                               milestones=[i * (int)(len(train_set)/params.batch_size + 1) for i in params.milestones])
    # print("optimizer.param_groups[0]['lr'] after LR_SCHEDULER", OPTIMIZER.param_groups[0]['lr'])

    # augmentation using data sampler
    TRAIN_LOADER = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=8,
                              collate_fn=data.Preprocessing.pad_packed_collate)
    TEST_LOADER = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=8,
                             collate_fn=data.Preprocessing.pad_packed_collate)
    VAL_LOADER = DataLoader(val1_set, batch_size=params.batch_size, shuffle=True, num_workers=8,
                            collate_fn=data.Preprocessing.pad_packed_collate)

    # Train model
    if params.train:
        train(MODEL, CRITERION, OPTIMIZER, LR_SCHEDULER, TRAIN_LOADER, VAL_LOADER, LEN_VAL1_SET)

    # eventually save model and optimizer state
    if params.save:
        torch.save(MODEL.state_dict(), '{0}/netRCNN.pth'.format(log_dir))
        print("Network saved at location %s" % log_dir)
        torch.save(OPTIMIZER.state_dict(), '{0}/optimizer_state.pth'.format(log_dir))
        print("Optimizer state saved at location %s" % log_dir)

    # Test model
    test(MODEL, CRITERION, TEST_LOADER, LEN_TEST_SET)
    del MODEL
