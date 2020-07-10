import numpy as np
from skimage import io as img_io
import os
import torch

line_gt = '/media/vn_nguyen/hdd/hux/IAM/lines.txt'
line_img = '/media/vn_nguyen/hdd/hux/IAM/lines/'

'''
Data utils for IAM dataset
'''

def gather_iam_line():
    '''
    Read given dataset IAM from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    gtfile = line_gt
    root_path = line_img
    gt = []
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]

            if (info[1] != 'ok'): # if the line is not properly segmented
                continue

            img_path = '/'.join(pathlist)
            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))
    return gt

def iam_main_loader():
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''

    line_map = gather_iam_line()

    data = []
    for i, (img_path, transcr) in enumerate(line_map):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(line_map), 100. * i / len(line_map)))

        try:
            img = img_io.imread(img_path + '.png')
            img = 1 - img.astype(np.float32) / 255.0
        except:
            continue

        data += [(img, transcr.replace("|", " "))]
    return data



# test the functions
if __name__ == '__main__':
    data = iam_main_loader()
    print(data[10][0])
    print(data[10][0].shape)
    print(data[10][0].sum())
    print("Success")