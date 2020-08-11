import numpy as np
from skimage.transform import resize
from skimage import transform

from skimage import io as img_io
from skimage.color import rgb2gray
import torch.nn.functional as F

import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

# ------------------------------------------------
'''
In this block : collate function for DataLoader
'''


def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(img, gt)].
             img is a FloatTensor
             gt is a str
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """
    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        # sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels = [labels]
    if len(batch) > 1:
        sigs, labels, lengths = zip(
            *[(a, b, a.size(2)) for (a, b) in sorted(batch, key=lambda x: x[0].size(2), reverse=True)])
        n_channel, n_feats, max_len = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(n_channel, n_feats, max_len - s.size(2))), 2) if s.size(2) != max_len else s for s in
                sigs]
        sigs = torch.stack(sigs, 0)
    packed_batch = pack(sigs, lengths, batch_first=True)
    return packed_batch, labels

# ------------------------------------------------
'''
In this block : Image preprocessing
    img_resize
    img_affine
    img_centered
    img_shear
    img_deslant
    -> preprocessing
'''


def img_resize(img, height=None, width=None, keep_ratio=True):
    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale * img.shape[1])
        img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale * img.shape[0])
        img = resize(image=img, output_shape=(height, width)).astype(np.float32)
    # keep height-width ratio
    if keep_ratio:
        if height is not None and width is not None:
            scale_h = float(height) / float(img.shape[0])
            scale_w = float(width) / float(img.shape[1])
            if scale_h * float(img.shape[1]) <= width:  # padding in width
                w = int(scale_h * float(img.shape[1]))
                img = resize(image=img, output_shape=(height, w)).astype(np.float32)
                img = np.pad(img, ((0, 0), (0, width - w)), 'constant', constant_values=0)
            else:  # padding in height
                h = int(scale_w * img.shape[0])
                img = resize(image=img, output_shape=(h, width)).astype(np.float32)
                img = np.pad(img, ((0, height - h), (0, 0)), 'constant', constant_values=0)
    else:
        img = resize(image=img, output_shape=(height, width)).astype(np.float32)
    return img


def img_affine(img):
    return img


def img_centered(img):
    img_r = img_resize(img, height=img.shape[0]-6, width=img.shape[1]-20)
    img_c = np.pad(img_r, ((0, 0), (10, 10)), 'constant', constant_values=0)
    img_c = np.pad(img_c, ((3, 3), (0, 0)), 'constant', constant_values=0)
    assert img_c.shape[0] == img.shape[0]
    assert img_c.shape[1] == img.shape[1]
    return img_c


def img_shear(img, ang):
    tform = transform.AffineTransform(shear=ang)
    tf_img = transform.warp(img, tform, order=1, preserve_range=True, mode='constant')
    # tf_img = tf_img.astype(np.float32) / 255.0
    return tf_img


def img_deslant(img):
    alphaVals = np.arange(-0.5, 0.5, 0.1)

    best_alpha = -0.5
    max_sum_alpha = 0

    for alpha in alphaVals:
        # print("alpha =", alpha)
        sum_alpha = 0
        sheared_image: np.ndarray = img_shear(img, alpha)
        fg = sheared_image  # == 0
        fg = (fg > 0.4).astype(np.float32)
        # print("fg =", fg)
        # print("fg.sum() =", fg.sum())
        h_alpha = fg.sum(axis=0)  # stroke pixel number in one col
        # print("h_alpha =", h_alpha)
        for col in range(fg.shape[1]):
            indexes = np.nonzero(fg[:, col])[0]
            if len(indexes) != 0:
                d_y_alpha = np.max(indexes) - np.min(indexes)
                if d_y_alpha == h_alpha[col]:  # one continue stroke in this line
                    # print("d_y_alpha =", d_y_alpha)
                    # print("h_alpha[col] =", h_alpha[col])
                    sum_alpha += h_alpha[col] ** 2
        if sum_alpha > max_sum_alpha:
            # print("sum_alpha =", sum_alpha, " aplha =", alpha)
            max_sum_alpha = sum_alpha
            best_alpha = alpha  # with more like one continue stroke in line, more likely a deslant image
    # print("best_alpha =", best_alpha)
    result = img_shear(img, best_alpha)
    return result


def preprocessing(img, data_size=(32, None), affine=False, centered=False, deslant=False, keep_ratio=True):
    if centered:
        img = img_centered(img)
    if deslant:
        img = img_deslant(img)
    if affine:
        img = img_affine(img)
    img = img_resize(img, height=data_size[0], width=data_size[1], keep_ratio=keep_ratio)

    return img
