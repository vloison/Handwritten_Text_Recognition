# Imports for myDataset
import data.IAM_dataset
import data.ICFHR2014_dataset
import data.synlines_dataset
from data.Preprocessing import preprocessing, pad_packed_collate
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
# Imports for lmdb
import lmdb
import six
import sys
from PIL import Image
from skimage import io as img_io
from skimage import draw
import linecache
import os



class myDataset(Dataset):
    def __init__(self, data_type='IAM', set='train', data_size=(32, None),
                 affine=False, centered=False, deslant=False, data_aug=False, keep_ratio=True, enhance_contrast=False):
        self.data_size  = data_size
        self.affine     = affine
        self.centered   = centered
        self.deslant    = deslant
        self.keep_ratio = keep_ratio
        self.enhance_contrast = enhance_contrast
        self.data_aug   = data_aug
        if data_type == 'IAM':
            self.data = data.IAM_dataset.iam_main_loader(set)
        elif data_type == 'ICFHR2014':
            self.data = data.ICFHR2014_dataset.icfhr2014_main_loader(set)
        elif data_type == 'synlines':
            self.data = data.synlines_dataset.synlines_main_loader(set)
        else:
            print("data_type unknown. Valid values are 'IAM' or 'ICFHR2014' or 'synlines'.")

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # Other possible data augmentation
                # transforms.RandomAffine(degrees=(-3, 3), translate=(0, 0.2), scale=(0.9, 1),
                #                         shear=5, resample=False, fillcolor=255),
                transforms.RandomAffine(degrees=(-2, 2), translate=(0, 0), scale=(0.9, 1),
                                        shear=5, resample=False, fillcolor=0),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=255)
            ]
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item][0]
        gt  = self.data[item][1]

        # data pre-processing
        img = preprocessing(img, self.data_size, affine=self.affine,
                            centered=self.centered, deslant=self.deslant, keep_ratio=self.keep_ratio,
                            enhance_contrast=self.enhance_contrast)

        # data augmentation
        if self.data_aug:
            img = torch.Tensor(img).float().unsqueeze(0)
            img = self.transform(img)
            img = transforms.ToTensor()(img).float()
        else:
            img = torch.Tensor(img).float().unsqueeze(0)

        return img, gt


class lmdbDataset(Dataset):

    def __init__(self, root='/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/BRNO_/lines/',
                 dataset='train.easy', data_size=(32, 400)):
        self.root = root + dataset

        # delete existing mdb if exists
        # path = ''.join(self.root + '/data.mdb')
        # if os.path.exists(path):
        #     os.remove(path)
        # path = ''.join(self.root + '/lock.mdb')
        # if os.path.exists(path):
        #     os.remove(path)

        self.env = lmdb.open(self.root.encode("utf8"), map_size=int(1e9), lock=False)
        self.dataset = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/BRNO_/' + dataset
        self.data_size = data_size

        linenum = len(open(self.dataset, 'rU').readlines())

        # with self.env.begin(write=True) as txn:
        #     # print(linenum)
        #     for i in range(linenum):
        #         line = linecache.getline(self.dataset, i+1).strip()
        #         img = 'image-%08d' % i
        #         label = 'label-%08d' % i
        #         txn.put(img.encode(), (root + line[:50]).encode())
        #         txn.put(label.encode(), line[51:].encode())

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
    test_set = myDataset(data_type='ICFHR2014', data_size=(32, 400), set='test', data_aug=True, keep_ratio=True)
    print("len(test_et_set) =", test_set.__len__())
    # # DRAW ONE LINE AT EACH PREDICTION
    # len_prediction = 99  # check value by running network.py
    # window_width = int(400 / len_prediction)
    # # Save images with one vertical line at each spot the network makes a prediction
    # for k in range(10):
    #     img = train_set[k][0]
    #     img = img.squeeze(0)
    #     for l in range(1, len_prediction+1):
    #         #rr, cc = draw.line(0, window_width*l, 31, window_width*l)
    #         #img[rr, cc] = 0.6
    #         img_io.imsave('/home/loisonv/images/train_set_IAM{0}.jpg'.format(k), img)

    # Show data augmentation
    # for k in range(10):
    #     img = test_set[k][0]
    #     img = img.squeeze(0)
    #     img_io.imsave('/home/loisonv/images/test_data_aug_tr0_{0}.jpg'.format(k), img)

    # augmentation using data sampler
    batch_size = 8
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=pad_packed_collate)
    for iter_idx, (img, gt) in enumerate(test_loader):
        print("img.size() =", img.data.size())
        print("gt =", gt)
        if iter_idx == 2:
            break