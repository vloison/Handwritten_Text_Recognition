import data.data_utils
from data.Preprocessing import preprocessing
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class myDataset(Dataset):
    def __init__(self, data_type = 'IAM', set = 'train', data_size=(32, None),
                 affine = False, centered = False, deslant = False, data_aug = False):
        self.data_size = data_size
        self.affine = affine
        self.centered = centered
        self.deslant = deslant
        if data_type == 'IAM':
            self.data = data.data_utils.iam_main_loader(set, data_aug)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item][0]
        gt = self.data[item][1]

        # data pre-processing
        img = preprocessing(img, self.data_size, affine=self.affine,
                            centered=self.centered, deslant=self.deslant)

        # data to tensor
        img = torch.Tensor(img).float().unsqueeze(0)

        return img, gt

import lmdb
import six
import sys
from PIL import Image
import linecache
import os

class lmdbDataset(Dataset):

    def __init__(self, root='/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/BRNO_/lines/',
                 dataset='train.easy', data_size=(32, 400)):
        self.root = root + dataset

        # delete existing mdb if exists
        path = ''.join(self.root + '/data.mdb')
        if os.path.exists(path):
            os.remove(path)
        path = ''.join(self.root + '/lock.mdb')
        if os.path.exists(path):
            os.remove(path)

        self.env = lmdb.open(self.root.encode("utf8"), map_size=int(1e9), lock=False)
        self.dataset = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/BRNO_/' + dataset
        self.data_size = data_size

        linenum = len(open(self.dataset, 'rU').readlines())

        with self.env.begin(write=True) as txn:
            # print(linenum)
            for i in range(linenum):
                line = linecache.getline(self.dataset, i+1).strip()
                img = 'image-%08d' % i
                label = 'label-%08d' % i
                txn.put(img.encode(), (root + line[:50]).encode())
                txn.put(label.encode(), line[51:].encode())

        if not self.env:
            print('cannot creat lmdb from %s' % (self.root))
            sys.exit(0)

        # with self.env.begin(write=False) as txn:
        #     nSamples = int(txn.get('num-samples'.encode('utf-8')))
        #     self.nSamples = nSamples

        self.nSamples = linenum

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples # len(self), 'index range error'
        # index += 1
        # line = linecache.getline(self.dataset, index).strip()
        with self.env.begin(write=False) as txn:
            img_key = 'image-%08d' % index
            # print(img_key)
            imgbuf = txn.get(img_key.encode('utf-8')).decode()
            # print(imgbuf)
            # buf = six.BytesIO()
            # buf.write(imgbuf)
            # buf.seek(0)
            try:
                img = Image.open(''.join(imgbuf)).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            img = 1 - np.asarray(img).astype(np.float32) / 255.0
            img = preprocessing(img, data_size=self.data_size)
            img = torch.Tensor(img).float().unsqueeze(0)


            label_key = 'label-%08d' % index
            label = txn.get(label_key.encode('utf-8')).decode()
            # label = line[52:]

        return (img, label)




# test above functions
if __name__ == '__main__':
    val1_set = myDataset(data_size=(32, None), set='val1')
    print("len(val1_set) =", val1_set.__len__())

    # augmentation using data sampler
    batch_size = 8
    val_loader = DataLoader(val1_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=data_utils.pad_packed_collate)
    for iter_idx, (img, gt) in enumerate(val_loader):
        print("img.size() =", img.data.size())
        print("gt =", gt)
        if iter_idx == 2:
            break