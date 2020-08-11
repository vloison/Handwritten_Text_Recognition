import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm


# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO ICFHR2014 DATASET ON SSD
LINE_GT_ICFHR2014 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/ICFHR2014/BenthamDatasetR0-GT/Transcriptions/'
LINE_IMG_ICFHR2014 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/ICFHR2014/BenthamDatasetR0-GT/Images/Lines/'
LINE_TRAIN_ICFHR2014 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/ICFHR2014/BenthamDatasetR0-GT/Partitions/TrainLines.lst'
LINE_TEST_ICFHR2014 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/ICFHR2014/BenthamDatasetR0-GT/Partitions/TestLines.lst'
LINE_VAL_ICFHR2014 = '/media/vn_nguyen/00520aaf-5941-4990-ae10-7bc62282b9d5/hux_loisonv/ICFHR2014/BenthamDatasetR0-GT/Partitions/ValidationLines.lst'

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
        if len(transcr) <= 100:
            img_path = root_img_path + line_id + '.png'
            line_map.append((img_path, transcr))
    return line_map


def icfhr2014_main_loader(set='train', data_aug=False):
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
            if set == 'train' and data_aug:  # augment data with shear
                tform = transform.AffineTransform(shear=np.random.uniform(-0.3, 0.3))
                inverted_img = util.invert(img)
                tf_img = transform.warp(inverted_img, tform, order=1, preserve_range=True, mode='constant')
            img = 1 - img
        except:
            print('exception raised')
            # continue
        # Add (image, transcr) to the list
        data += [(img, transcr)]
        if set == 'train' and data_aug:  # augment data with shear
            data += [(tf_img, transcr)]
    print("Reading done.")
    return data


if __name__ == '__main__':
    dataset = icfhr2014_main_loader('train')
    print(dataset[0][0].shape)

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