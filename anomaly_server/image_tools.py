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
    """Utility for loading image files into memory"""
    return map(imread, fns)

def is_different(a, b):
    """Does a quick raw pixel level check to determine if futher comparison is required."""
    mse = compare_mse(a, b)
    ssim = compare_ssim(a, b, multichannel=True)

    return mse > 30 and ssim > .8

def create_delta_image(a, b, fn="", save=False):
    """
    Generates a dimage

    A dimage is just an image representation of the difference
    between two images on each color channel. Two identical images
    will have a dimage of just black while completely opposite images
    will result in pure white.

    Parameters
    ----------
    a, b : numpy.array
        Input images
    fn : str
        Save filename
    save : bool
        True to save the dimage

    Returns
    -------
    numpy.array
        The dimage created
    """
    dimage = np.zeros_like(a, dtype=np.uint8)

    for c in range(3):
        dimage[:,:,c] = np.abs(np.subtract(a[:,:,c], b[:,:,c], dtype=np.int8))

    if save:
        imsave(fn, dimage)

    return dimage

def is_anomaly(anomaly_model, dimage):
    """Uses given keras model to predict the dimage"""
    return anomaly_model.predict(np.array([dimage]))[0,0] == 1

def create_crop_image(dimage, actual, fn="", save=False):
    """
    Generates a crop image

    A using the dimage, the important or anomalous part of the image
    is extracted and is used as a mask on the actual image where the motion
    takes place allowing for a cropped image of the anomaly in the frame
    to be created.

    Parameters
    ----------
    dimage, actual : numpy.array
        Input images
    fn : str
        Save filename
    save : bool
        True to save the crop image

    Returns
    -------
    numpy.array
        The crop image created
    """
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
