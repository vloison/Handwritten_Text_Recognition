import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm
import glob
import os


# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO SYNTHETIC LINES DATASET
synlines_path = '/home/hux/docExtractor/datasets/synlines_small/'
# ------------------------------------------------
'''
In this block : Data utils for synlines dataset
'''

def gather_syn_line(set='train'):
    '''
    Read given dataset IAM from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    # Load global variables
    root_img_path = synlines_path
    # Initialize result list
    line_map = []
    # Load line_id of the set
    data_set = root_img_path + set
    # print("Cannot find this dataset. Valid values for set are 'train', 'test' or 'val'. ")
    # Build list
    paths = glob.glob(os.path.join(data_set, '*.jpg'))
    paths.sort()
    for path in paths:
        img_path = path
        gt_path = img_path[:-4] + '_ocr.txt'
        transcr_file = open(gt_path, 'r')
        transcr = transcr_file.read()
        # Get rid of \n character
        # transcr = transcr.strip()
        # Ignore trancripts that have length >100 to avoid nan loss
        if len(transcr) < 95:
            line_map.append((img_path, transcr))
    return line_map


def synlines_main_loader(set='train'):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''
    # Load the pairs : (image path, ground truth text)
    line_map = gather_syn_line(set=set)
    print(len(line_map))
    # Initialize results list
    data = []
    print("Reading synlines dataset..")
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        # Load image and its eventual transformation
        try:
            img = img_io.imread(img_path, as_gray=True)
            img = 1 - img
        except:
            print('exception raised')
            # continue
        # Add (image, transcr) to the list
        data += [(img, transcr)]
        # if set == 'train' and data_aug:  # augment data with shear
        #     data += [(tf_img, transcr)]
    print("Reading done.")
    return data


if __name__ == '__main__':
    dataset = synlines_main_loader('train')
    print(dataset[0][0].shape)
    print(dataset[0][0])
    print(dataset[31][0])
    print(dataset[0][1])
    print(dataset[31][1])
    print(dataset[61][1])
    print(dataset[131][1])

    # (img_path, transcr) = gather_icfhr2014_line('train')[0]
    # img = img_io.imread(img_path, as_gray=True)
    # tform = transform.AffineTransform(shear=np.random.uniform(-0.3, 0.3))
    # inverted_img = util.invert(img)
    # tf_img = transform.warp(inverted_img, tform, order=1, preserve_range=True, mode='constant')
    # img_io.imsave('/home/loisonv/tf_image.jpg', tf_img)
    # print(img.shape)
    # print(np.max(img))
    # print(np.min(img))
    # img_io.imsave('/home/loisonv/before_trans_as_gray_ioimsave2.jpg', img)
    # img = 1 - img
    # print(np.max(img))
    # print(np.min(img))
    # img_io.imsave('/home/loisonv/after_trans_as_gray_ioimsave2.jpg', img)

    # bla = np.loadtxt(LINE_TEST_ICFHR2014, dtype=str)
    # print(bla[0])
    # ident = '002_080_001_01_01'
    # transcr = np.loadtxt(LINE_GT_ICFHR2014 + ident + '.txt', dtype=str)
    # print(transcr)

    # blabla = gather_icfhr2014_line('test')
    # print(len(blabla))