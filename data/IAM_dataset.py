import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm
from params import *

params = BaseOptions().parser()

# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO HDD
# line_gt = '/media/vn_nguyen/hdd/hux/IAM/lines.txt'
# line_img = '/media/vn_nguyen/hdd/hux/IAM/lines/'
# line_train = '/media/vn_nguyen/hdd/hux/IAM/split/trainset.txt'
# line_test = '/media/vn_nguyen/hdd/hux/IAM/split/testset.txt'
# line_val1 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset1.txt'
# line_val2 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset2.txt'


# PATH TO IAM DATASET ON SSD
line_gt = params.tr_data_path + 'IAM/lines.txt'
line_img = params.tr_data_path + 'IAM/lines/'
line_train = params.tr_data_path + 'IAM/split/trainset.txt'
line_test = params.tr_data_path + 'IAM/split/testset.txt'
line_val1 = params.tr_data_path + 'IAM/split/validationset1.txt'
line_val2 = params.tr_data_path + 'IAM/split/validationset2.txt'


# ------------------------------------------------
'''
In this block : Data utils for IAM dataset
'''


def gather_iam_line(set='train'):
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
    elif set == 'val':
        data_set = np.loadtxt(line_val1, dtype=str)
    elif set == 'val2':
        data_set = np.loadtxt(line_val2, dtype=str)
    else:
        print("Cannot find this dataset. Valid values for set are 'train', 'test', 'val' or 'val2'.")
        return
    gt = []
    print("Reading IAM dataset...")
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
    print("Reading done.")
    return gt


def iam_main_loader(set='train'):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''

    line_map = gather_iam_line(set)

    data = []
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        try:
            img = img_io.imread(img_path + '.png')
            # if set == 'train' and data_aug:  # augment data with shear
            #     tform = transform.AffineTransform(shear=np.random.uniform(-0.3, 0.3))
            #     inverted_img = util.invert(img)
            #     tf_img = transform.warp(inverted_img, tform, order=1, preserve_range=True, mode='constant')
            #     tf_img = tf_img.astype(np.float32) / 255.0
            img = 1 - img.astype(np.float32) / 255.0
            # img = img.astype(np.float32) / 255.0
        except:
            continue

        data += [(img, transcr.replace("|", " "))]
        # if set == 'train' and data_aug:  # augment data with shear
        #     data += [(tf_img, transcr.replace("|", " "))]
    return data

# ------------------------------------------------


# test the functions
if __name__ == '__main__':
    (img_path, transcr) = gather_iam_line('train')[0]
    img = img_io.imread(img_path + '.png')
    print(img.shape)
    print(img)

    data = iam_main_loader(set='train')
    print("length of trainset:", len(data))
    print(data[10][0].shape)

    data = iam_main_loader(set='test')
    print("length of testset:", len(data))

    data = iam_main_loader(set='val')
    print("length of val set:", len(data))
    print("Success")
