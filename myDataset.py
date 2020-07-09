import data_utils
import Preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class myDataset(Dataset):
    def __init__(self, data_type = 'IAM', data_size=(32, None),
                 affine = False, centered = False):
        self.data_size = data_size
        self.affine = affine
        self.centered = centered
        if data_type == 'IAM':
            self.data = data_utils.iam_main_loader()

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
    full_dataset = myDataset(data_size=(32, 100)) # if set data_size = (32, None), we need to set batch_size = 1
    print("len(full_dataset) =", full_dataset.__len__())

    # split the data into training set and test set
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    print("train_size =", train_size)
    print("test_size =", test_size)
    print("train_size + test_size =", train_size+test_size)
    train_set, test_set = random_split(full_dataset, [train_size, test_size])
    print("len(train_set) =", train_set.__len__())
    print("len(test_set) =", test_set.__len__())

    # augmentation using data sampler
    batch_size = 5
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    for iter_idx, (img, gt) in enumerate(train_loader):
        print("img.size =", img.size())
        print("gt =", gt)
        break