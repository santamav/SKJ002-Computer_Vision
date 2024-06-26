#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy import ndimage as filters
import scipy.signal
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time

sys.path.append("Lab1/") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu

# ----------------------
# Fourier Transform
# ----------------------

def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))  # perform also the shift to have lower frequencies at the center


def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT

def exercici_1(abs_mag, log_mag, phase):
    print(f'Magnitude Abs min:{np.min(abs_mag)} max:{np.max(abs_mag)}')
    print(f'Magnitude Log min:{np.min(log_mag)} max:{np.max(log_mag)}')
    
    fig, axs = plt.subplots(1, 3)
    axs[0].boxplot([np.min(abs_mag),np.max(abs_mag)])
    axs[0].set_title('Magnitude Abs')
    axs[1].boxplot([np.min(log_mag),np.max(log_mag)])
    axs[1].set_title('Magnitude Log')
    axs[2].hist(phase)
    axs[2].set_title('Phase')

def testFT(im, params=None):
    ft = FT(im)
    #print(ft.shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    exercici_1(np.absolute(ft), magnitude, phase)
    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]

# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)


# apply average filter in the spatial domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))  # both '*' and multiply() perform elementwise product


def testConvTheo(im, params=None):
    filterSize = params['filterSize']

    # image filtered with a convolution in spatial domain
    imFiltSpace = averageFilterSpace(im, filterSize)

    # image filtered in frequency domain
    imFiltFreq = averageFilterFrequency(im, filterSize)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)

    return [imFiltSpace, imFiltFreq]
False
# -----------------------------------
# Comvolution theorem for gaussian filter
# -----------------------------------
def gaussianFilter(n, sigma):
    x = np.linspace(-n//2, n//2, n)
    y = np.linspace(-n//2, n//2, n)
    xv, yv = np.meshgrid(x, y)
    filterMask = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    return filterMask / np.sum(filterMask)

def gaussianFilterSpace(im, n, sigma):
    return filters.convolve(im, gaussianFilter(n, sigma))


def gaussianFilterFrequency(im, n, sigma):
    filterMask = gaussianFilter(n, sigma)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return FT(im) * FT(filterBig) # both '*' and multiply() perform elementwise product


def testConvTheoGaussian(im, params=None):
    sizes = params['n']
    sigmas = params['sigma']
    imgs = []
    
    for n in sizes:
        for sigma in sigmas:
                # I, IFT(FT(I).FT(M), |FT(M)|, |FT(I)*FT(M)|
                # Fourier transform of the filter
                filterFt = FT(gaussianFilter(n, sigma))

                # image filtered with a convolution in spatial domain
                imFiltSpace = gaussianFilterSpace(im, n, sigma)

                # image filtered in frequency 
                ft_im = gaussianFilterFrequency(im, n, sigma)
                imFiltFreq = np.absolute(IFT(ft_im))
                ## Then, paste the small mask at the center using the sizes computed before as an aid
                filterBig = np.zeros_like(im, dtype=float)
                ## First, get sizes
                w, h = filterFt.shape
                w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
                W, H = filterBig.shape
                W2, H2 = W / 2, H / 2  # half width and height of the "big" mask
                filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterFt
                ft_m = np.absolute(IFT(filterBig))
                ft_im = np.log(np.absolute(ft_im))
                
                #imgs.append(imFiltFreq)
                #imgs.append(ft_m)
                imgs.append(ft_im)
    
    return imgs 
            
#------------------------------------
# Exercise 4
#------------------------------------

def my_mask(n):
    mask = np.triu(np.ones((n,n)), 0) - np.tril(np.ones((n,n)), 0)
    print(mask)
    return mask

def my_filter(im, mask):
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)
    # Now, place filter (the "small" filter mask) at the center of the "big" filter
    
    ## First, get sizes
    w, h = mask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    
    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W/2 - w/2):int(W/2 + w/2), int(H/2 - h/2):int(H/2 + h/2)] = mask
    
    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner
    
    # Finally, IFT of the element-wise product of the FT's
    im_filt = np.absolute(IFT(FT(im) * FT(filterBig)) ) # both '*' and multiply() perform elementwise product
    
    return im_filt

# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def bandPassFilter(shape, r=None, R=None):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        filter = distToCenter<r # same as np.less(distToCenter, r)
    elif r is None:  # high-pass filter assumed
        filter = distToCenter>R # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = R, r  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter<R, distToCenter>r)
    filter = filter.astype('float')  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap='gray')
        plt.show()
        plt.title("The filter in the frequency domain")
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params['r'], params['R']
    filterFreq = bandPassFilter(im.shape, r, R)  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(filterFreq)  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image


# -----------------
# Test image files
# -----------------
path_input = './Lab3/imgs-P3/'
path_output = './Lab3/imgs-out-P3/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.jpg")
else:
    files = [path_input + 'einstein.jpg']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testConvTheoGaussian']#['testConvTheoGaussian']
    #tests = ['testFT', 'testConvTheo', 'testBandPassFilter']
else:
    tests = ['testFT']
    tests = ['testConvTheo']
    tests = ['testBandPassFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testFT': '2D Fourier Transform',
             'testConvTheo': 'Convolution Theorem (tested on mean filter)',
             'testBandPassFilter': 'Frequency-based filters ("high/low/band-pass")',
             'testConvTheoGaussian': 'Convolution Theorem (tested on gaussian filter)'
             }

bSaveResultImgs = False

testsUsingPIL = []  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            if test is "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"
                pltTitles = None
            elif test is "testConvTheo":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"
                pltTitles = None
            elif test is "testConvTheoGaussian":
                params = {}
                params['sigma'] = [16]
                params['n'] = [3, 7, 15]
                subTitle = ": I, IFT(FT(I).FT(M)"
                #pltTitles = ['im', 'IFT(FT(I)*FT(M))', '|FT(M)|', '|FT(I)*FT(M)|']
                pltTitles = ['im','n=3', 'n=7', 'n=15']
                m = None
                n = 4
            else:
                params = {}
                r,R = 5,None # for low-pass filter
                # 5,30 for band-pass filter
                # None, 30 for high-pass filter
                params['r'], params['R'] = r,R
                # let's assume r and R are not both None simultaneously
                if r is None:
                    filter="high pass" + " (R=" + str(R) + ")"
                elif R is None:
                    filter="low pass" + " (r=" + str(r) + ")"
                else:
                    filter="band pass" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", " + filter + " filter"
                pltTitles = None

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
            print("# images", len(outs_np))
            print(len(outs_np))

            #vpu.showInGrid(outs_np, n=n, m=m, title=nameTests[test] + subTitle, subtitles=pltTitles)
            vpu.showInGrid([im] + outs_np, n=n, m=m, title=nameTests[test] + subTitle, subtitles=pltTitles)
        
def doFiltervsSpatial():
    sizes = [3, 7, 15, 31, 63]
    sigma = 1.2
    
    for file in files:
        im_pil = Image.open(file).convert('L')
        im = np.array(im_pil)
        
        filterSpaceTimes = []
        filterFreqTimes = []
        x_names = []
        fig, axs = plt.subplots()
        
        for size in sizes:
            x_names.append(f'{size}')
            t0 = time.time()
            imFiltSpace = gaussianFilterSpace(im, size, sigma)
            t1 = time.time()
            imFiltFreq = gaussianFilterFrequency(im, size, sigma)
            t2 = time.time()
            print(f'Filter size: {size}x{size} - Time space: {t1-t0} - Time freq: {t2-t1}')
            filterSpaceTimes.append(t1-t0)
            filterFreqTimes.append(t2-t1)
            #vpu.showInGrid([im, imFiltSpace, imFiltFreq], n=1, m=3, title=f'Filter size: {size}x{size}', subtitles=['Original', 'Spatial', 'Frequency'])
        # Plot times
        axs.plot(x_names, filterSpaceTimes, label='Spatial')
        axs.plot(x_names, filterFreqTimes, label='Frequency')
        # Set title
        axs.set_title(f'Filter vs Spatial: {im.shape[0]}x{im.shape[1]}')
        axs.legend()
        plt.show()
    
def doTestMyFilter():
    for file in files:
        im_pil = Image.open(file).convert('L')
        im = np.array(im_pil)
        mask = my_mask(3)
        im_filt = my_filter(im, mask)
        vpu.showInGrid([im, im_filt], n=2, m=1, title='My filter', subtitles=['Original', 'Filtered'])
        
def doExercise5():
    lmbdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    imgs = []
    # Load images
    im1 = np.array(Image.open(path_input + 'stp1.gif').convert('L'))
    im2 = np.array(Image.open(path_input + 'stp2.gif').convert('L'))    
    # Transform of the images
    ft_im1 = FT(im1)
    ft_im2 = FT(im2)
    
    for lmbda in lmbdas:
        # Combination of the images
        ft_prod = (lmbda * ft_im1) + ((1 - lmbda) * ft_im2)
        # Inverse transform of the combination
        im_comb = np.absolute(IFT(ft_prod))
        imgs.append(im_comb)
    
    vpu.showInGrid(imgs, title='Exercise 5', m=1, n=len(lmbdas), subtitles=[f'Lambda: {lmbda}' for lmbda in lmbdas])
    
def doExercise6():
    lmda = 0.3
    imgs = []
    # Load images
    im1 = np.array(Image.open(path_input + 'stp1.gif').convert('L'))
    im2 = np.array(Image.open(path_input + 'stp2.gif').convert('L'))
    # Combination of the images in the spatial domain
    im_comb = (lmda * im1) + ((1 - lmda) * im2)    
    # Transform of the combination
    im_comb_esp = np.absolute(FT(im_comb))
    imgs.append(im_comb_esp)
    
    # Combination of the images in the frequency domain
    ft_im1 = FT(im1)
    ft_im2 = FT(im2)
    ft_comb = (lmda * ft_im1) + ((1 - lmda) * ft_im2)
    im_comb_freq = np.absolute(IFT(ft_comb))
    ft_comb = np.absolute(ft_comb)
    imgs.append(ft_comb)
    
    #Empty white image
    space = np.zeros_like(im1)
    # Display images
    vpu.showInGrid(imgs, title='Exercise 6', m=1, n=3, subtitles=['F(I1)*F(I2)', 'F(I1+I2)'])
    
        # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(im_comb_esp[margin:-margin, margin:-margin] - ft_comb[margin:-margin, margin:-margin], 2) / np.prod(ft_comb.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)
    

if __name__ == "__main__":
    #doTests()
    #doFiltervsSpatial()
    #doTestMyFilter()
    #doExercise5()
    doExercise6()
