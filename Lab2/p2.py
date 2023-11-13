#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time

sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu


# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # Now, im is a PIL image (not a NumPy array)
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use (*)
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # number of pixels
    m = int(math.floor(percent * N / 100.0)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # generate m random positions in the 1D array (index 0 to N-1)

    # generate m random S/P values (salt and pepper in the same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (0=salt and 1=pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape) # (*) here is where we use the shape that we saved earlier

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------

def addGaussianNoise(im, sd):
    if im.mode == 'L':
        im = np.array(im)
        result = im + np.random.normal(loc=0, scale=sd, size=im.shape)
        Image.fromarray(result).show()
        return result
    
    #if the mode is RGB, do the calculations for each band
    im_array = np.array(im)
    im_array[:,:,0] += np.clip(np.random.normal(loc=0, scale=sd, size=(im_array.shape[0], im_array.shape[1])),0,255).astype("uint8")
    im_array[:,:,1] += np.clip(np.random.normal(loc=0, scale=sd, size=(im_array.shape[0], im_array.shape[1])),0,255).astype("uint8")
    im_array[:,:,2] += np.clip(np.random.normal(loc=0, scale=sd, size=(im_array.shape[0], im_array.shape[1])),0,255).astype("uint8")
    return im_array
    
def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        print('testing sigma:', sigma)
        imgs.append(addGaussianNoise(im, sigma))
        print(len(imgs))
    return imgs

# -------------------------
# Average (or mean) filter
# -------------------------

def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask)) # can you think of any alternative for np.sum(mask)?
    return filters.convolve(im, mask)

#--------------------------
# Separable by axis Average (mean) filter
#--------------------------
def separableAverageFilter(im, filterSize):
    result = im.copy()
    mask = np.ones(filterSize)
    mask = np.divide(mask, np.sum(mask))
    #Filter the image in the two axis
    result = filters.convolve(result, mask.reshape(-1,1))
    result = filters.convolve(result, mask.reshape(1,-1))
    return result

def testSeparableAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(separableAverageFilter(im_dirty, filterSize))
    return imgs

def testAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(averageFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Gaussian filter
# -----------------
def gaussianFilterSep(im, sigma=5, n=16):
    gv1d = scipy.signal.gaussian(n, sigma)
    result = im.copy()
    #first dimension convolution
    result = filters.convolve(result, gv1d.reshape(-1, 1))
    #second dimension convolution
    result = filters.convolve(result, gv1d.T.reshape(1, -1))

    return result

def colorChannels_gaussianFilter(im, sigma =5):
    Image.fromarray(im).show()
    result = np.copy(im)
    result[:,:,0] = filters.gaussian_filter(im[:,:,0], sigma)
    result[:,:,1] = filters.gaussian_filter(im[:,:,1], sigma)
    result[:,:,2] = filters.gaussian_filter(im[:,:,2], sigma)
    Image.fromarray(result).show()
    return result

def old_gaussianFilter(im, sigma=5):
    result = im.copy()
    result = filters.gaussian_filter(result, sigma)
    return result

def gaussianFilter(im, sigma=5, n=16):
    # im is PIL image
    gv1d = scipy.signal.gaussian(n, sigma)
    #plt.imshow(gv1d.reshape(1, -1))
    gv2d = np.outer(gv1d, gv1d)
    #plt.imshow(gv2d)
    result = im.copy()
    result = filters.convolve(result, gv2d)
    return result


def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            #initial_time = time.time()
            #imgs.append(gaussianFilter(im_dirty, filterSize))
            #print(f'2D gasussina Filter: {time.time()-initial_time}')
            #initial_time = time.time()
            imgs.append(gaussianFilter(im_dirty, filterSize))
            #print(f'Separated gaussian filter: {time.time()-initial_time}')
    return imgs


def gaussianFilterEx3(im_clean, params):
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            im_dirty = addGaussianNoise(im_clean, sigma)
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilter(im_dirty, filterSize))
            imgs.append(old_gaussianFilter(im_dirty, filterSize))
    return imgs

# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs


# --------------------------
# Quotient Image
# --------------------------
def quotientImage(im, sigma):
    blurred_im = addGaussianNoise(im, sigma)
    result = im.copy()
    result = result/blurred_im
    plt.imshow(result)
    return result

def testQuotientImage(im, params):
    imgs = []
    for sigma in params['sd_gauss_noise']:
        imgs.extend((np.array(im), quotientImage(im, sigma)))
    return imgs

# -----------------
# Test image files
# -----------------

path_input = './Lab2/imgs-P2/'
path_output = './Lab2/imgs-out-P2/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*")
else:
    files = [path_input + 'lena512.pgm']  # lena256.pgm, lena512.pgm, peppers.ppm

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilter', 'testSeparableAverageFilter', 'testGaussianFilter', 'testMedianFilter']
bAllTests = False
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['gaussianFilterEx3']#['testAverageFilter', 'testSeparableAverageFilter', 'testGaussianFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Gaussian noise',
             'testSandPNoise': 'Salt & Pepper noise',
             'testAverageFilter': 'Mean filter',
             'testGaussianFilter': 'Gaussian filter',
             'testMedianFilter': 'Median filter',
             'testSeparableAverageFilter':'Separable Average',
             'testQuotientImage':'Quotient Image',
             'gaussianFilterEx3':'Gaussian Filter Ex3'}

bSaveResultImgs = False

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [20]#[3, 5, 10]  # standard deviation (for the [0,255] range) for Gaussian noise


# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['gaussianFilterEx3', 'testSandPNoise', 'testGaussianNoise', 'testGaussianFilter']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile)
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            if test == "testGaussianNoise":
                params = gauss_sigmas_noise
                subTitle = r", $\sigma$: " + str(params)
            elif test == "testSandPNoise":
                params = percentagesSandP
                subTitle = ", %: " + str(params)
            elif test == "testAverageFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testSeparableAverageFilter":
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testMedianFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testGaussianFilter":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)
            elif test == "gaussianFilterEx3":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)

            elif test == 'testQuotientImage':
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                subTitle = f", quotient image"
            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                print("num images", len(outs_np))
            print(len(outs_np))
            # display original image, noisy images and filtered images
            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)

if __name__ == "__main__":
    doTests()
