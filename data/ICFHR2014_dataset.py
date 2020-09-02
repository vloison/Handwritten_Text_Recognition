import numpy as np
from skimage import io as img_io
from tqdm import tqdm
from params import *

params = BaseOptions().parser()

# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO ICFHR2014 DATASET ON SSD
LINE_GT_ICFHR2014 = params.tr_data_path + 'ICFHR2014/BenthamDatasetR0-GT/Transcriptions/'
LINE_IMG_ICFHR2014 = params.tr_data_path + 'ICFHR2014/BenthamDatasetR0-GT/Images/Lines/'
LINE_TRAIN_ICFHR2014 = params.tr_data_path + 'ICFHR2014/BenthamDatasetR0-GT/Partitions/TrainLines.lst'
LINE_TEST_ICFHR2014 = params.tr_data_path + 'ICFHR2014/BenthamDatasetR0-GT/Partitions/TestLines.lst'
LINE_VAL_ICFHR2014 = params.tr_data_path + 'ICFHR2014/BenthamDatasetR0-GT/Partitions/ValidationLines.lst'

# ------------------------------------------------
'''
In this block : Data utils for ICFHR2014 dataset
'''


def gather_icfhr2014_line(set='train'):
    '''
    Read given dataset IAM from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    # Load global variables
    root_img_path = LINE_IMG_ICFHR2014
    # Initialize result list
    line_map = []
    # Load line_id of the set
    if set == 'train':
        data_set = np.loadtxt(LINE_TRAIN_ICFHR2014, dtype=str)
    elif set == 'test':
        data_set = np.loadtxt(LINE_TEST_ICFHR2014, dtype=str)
    elif set == 'val':
        data_set = np.loadtxt(LINE_VAL_ICFHR2014, dtype=str)
    else:
        print("Cannot find this dataset. Valid values for set are 'train', 'test' or 'val'. ")
        return
    # Build list
    for line_id in data_set:
        transcr_file = open(LINE_GT_ICFHR2014 + line_id + '.txt', 'r')
        transcr = transcr_file.read()
        # Get rid of \n character
        transcr = transcr.strip()
        # Ignore trancripts that have length >100 to avoid nan loss
        if len(transcr) < 100:
            img_path = root_img_path + line_id + '.png'
            line_map.append((img_path, transcr))
    return line_map


def icfhr2014_main_loader(set='train'):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''
    # Load the pairs : (image path, ground truth text)
    line_map = gather_icfhr2014_line(set=set)
    # Initialize results list
    data = []
    print("Reading ICFHR2014 dataset..")
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        # Load image and its eventual transformation
        try:
            img = img_io.imread(img_path, as_gray=True)
            #img = 1 - img
        except:
            print('exception raised')
            continue
        # Add (image, transcr) to the list
        data += [(img, transcr)]

    print("Reading done.")
    return data


if __name__ == '__main__':
    dataset = icfhr2014_main_loader('test')
    print(dataset[0][0].shape)
    for k in range(1):
        (img_path, transcr) = gather_icfhr2014_line('test')[k]
        img = img_io.imread(img_path, as_gray=True)

        img_io.imsave('/home/loisonv/images/original_image_test{0}.jpg'.format(k), img)
        img = 1 - img
        print('transformation')
        print('max(img)', np.max(img))
        print('min(img)', np.min(img))
        img_io.imsave('/home/loisonv/images/after_trans_test{0}.jpg'.format(k), img)
