import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm
import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

# PATH TO HDD
# line_gt = '/media/vn_nguyen/hdd/hux/IAM/lines.txt'
# line_img = '/media/vn_nguyen/hdd/hux/IAM/lines/'
# line_train = '/media/vn_nguyen/hdd/hux/IAM/split/trainset.txt'
# line_test = '/media/vn_nguyen/hdd/hux/IAM/split/testset.txt'
# line_val1 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset1.txt'
# line_val2 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset2.txt'

# PATH TO SSD
line_gt = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/lines.txt'
line_img = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/lines/'
line_train = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/split/trainset.txt'
line_test = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/split/testset.txt'
line_val1 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/split/validationset1.txt'
line_val2 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/IAM/split/validationset2.txt'




'''
Data utils for IAM dataset
'''

def gather_iam_line(set = 'train'):
    '''
    Read given dataset IAM from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    gtfile = line_gt
    root_path = line_img
    if set == 'train':
        data_set = np.loadtxt(line_train, dtype=str)
    elif set == 'test':
        data_set = np.loadtxt(line_test, dtype=str)
    elif set == 'val1':
        data_set = np.loadtxt(line_val1, dtype=str)
    elif set == 'val2':
        data_set = np.loadtxt(line_val2, dtype=str)
    else:
        print('Cannot find this dataset')
        return
    gt = []
    print("Read IAM dataset...")
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            line_name = pathlist[-1]
            if (info[1] != 'ok') or (line_name not in data_set):  # if the line is not properly segmented
                continue
            img_path = '/'.join(pathlist)
            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))
    return gt


def iam_main_loader(set='train', data_aug=False):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''

    line_map = gather_iam_line(set)

    data = []
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        try:
            img = img_io.imread(img_path + '.png')
            if set == 'train' and data_aug:  # augment data with shear
                tform = transform.AffineTransform(shear=np.random.uniform(-0.3, 0.3))
                inverted_img = util.invert(img)
                tf_img = transform.warp(inverted_img, tform, order=1, preserve_range=True, mode='constant')
                tf_img = tf_img.astype(np.float32) / 255.0
            img = 1 - img.astype(np.float32) / 255.0
        except:
            continue

        data += [(img, transcr.replace("|", " "))]
        if set == 'train' and data_aug:  # augment data with shear
            data += [(tf_img, transcr.replace("|", " "))]
    return data


def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(img, gt)].
             img is a FloatTensor
             gt is a str
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """
    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        # sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels = [labels]
    if len(batch) > 1:
        sigs, labels, lengths = zip(
            *[(a, b, a.size(2)) for (a, b) in sorted(batch, key=lambda x: x[0].size(2), reverse=True)])
        n_channel, n_feats, max_len = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(n_channel, n_feats, max_len - s.size(2))), 2) if s.size(2) != max_len else s for s in
                sigs]
        sigs = torch.stack(sigs, 0)
    packed_batch = pack(sigs, lengths, batch_first=True)
    return packed_batch, labels



# test the functions
if __name__ == '__main__':
    data = iam_main_loader(set = 'train')
    print("length of trainset:", len(data))
    print(data[10][0].shape)

    data = iam_main_loader(set='test')
    print("length of testset:", len(data))

    data = iam_main_loader(set='val1')
    print("length of val1 set:", len(data))
    print("Success")
