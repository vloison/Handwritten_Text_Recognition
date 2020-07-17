import numpy as np
from skimage.transform import resize
from skimage import io as img_io
from skimage.color import rgb2gray
import torch
import torch.nn.functional as F

def img_resize(img, height=None, width=None):
    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale * img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale * img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img

def img_affine(img):
    return img

def img_centered(word_img, tsize):
    return word_img


def preprocessing(input, data_size=(32, None), affine = False, centered = False):
    img = img_resize(input, height = data_size[0], width = data_size[1])
    if affine == True:
        img = img_affine(img)
    if centered == True:
        img = img_centered(img, (img.shape[0], int(1.2 * img.shape[1]) + 32))

    return img