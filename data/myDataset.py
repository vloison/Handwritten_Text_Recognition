import data.data_utils
import Preprocessing
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class myDataset(Dataset):
    def __init__(self, data_type = 'IAM', set = 'train', data_size=(32, None),
                 affine = False, centered = False, data_aug = False):
        self.data_size = data_size
        self.affine = affine
        self.centered = centered
        if data_type == 'IAM':
            self.data = data.data_utils.iam_main_loader(set, data_aug)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item][0]
        gt = self.data[item][1]

        # data pre-processing
        img = Preprocessing.preprocessing(img, self.data_size, affine=self.affine, centered=self.centered)

        # data to tensor
        img = torch.Tensor(img).float().unsqueeze(0)

        return img, gt


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