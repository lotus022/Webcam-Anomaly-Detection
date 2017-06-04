from skimage.exposure import rescale_intensity
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import label
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.io import imsave

import numpy as np

def load(*fns):
    return map(imread, fns)

def is_different(a, b):

    mse = compare_mse(a, b)
    ssim = compare_ssim(a, b, multichannel=True)

    return mse > 30 and ssim > .8

def create_delta_image(a, b, fn="", save=False):

    dimage = np.zeros_like(a, dtype=np.uint8)

    for c in range(3):
        dimage[:,:,c] = np.abs(np.subtract(a[:,:,c], b[:,:,c], dtype=np.int8))

    if save:
        imsave(fn, dimage)

    return dimage

def is_anomoly(anomoly_model, dimage):

    return anomoly_model.predict(np.array([dimage]))[0,0] == 1

def create_crop_image(dimage, actual, fn="", save=False):

    gray = gaussian(rgb2gray(dimage), sigma=100)
    gray = rescale_intensity(gray)

    blobs = gray > 2 * gray.mean()
    mask = label(blobs, background=0) >= 1

    bbox = np.argwhere(mask)
    (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
    crop = actual[ystart:ystop, xstart:xstop]

    if save:
        imsave(fn, crop)

    return crop
