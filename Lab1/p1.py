#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu

def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum() # cumulative distribution function (CDF) = cummulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cummulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf

def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def darkenImg(im,p=2):
    return (im ** float(p)) / (255 ** (p - 1)) # try without the float conversion and see what happens

def brightenImg(im,p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p)  # notice this NumPy function is different to the scalar math.pnp

def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*") #Changed from ~~files = glob.blob(path_input + "*.pgm")~~ so it reads all the files
else:
    files = [path_input + 'iglesia.pgm'] # iglesia,huesos, path_input + 'iglesia.pgm'

bAllTests = True
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg']
else:
    tests = ['testDarkenImg']#['testHistEq']#['testBrightenImg']
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk'}

def saveImage(imfile, im2, test): #Exercise 2 - Lab1
    dirname,basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    #print(dname,basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output+'//'+fname + suffixFiles[test] + fext)

def doRebrightenTest():
    for imfile in files:
        im = np.array(Image.open(imfile))
        darkIm = testDarkenImg(im)[0]
        brightIm = testBrightenImg(darkIm)[0]
        saveImage(imfile, darkIm, 'testDarkenImg')
        saveImage(imfile, brightIm, 'testBrightenImg')

def invertImg(im):
    return 255 - im

def checkboardIm(im, n, m):
    spacing_x = np.linspace(0, im.shape[0], n+1).astype(int)
    spacing_y = np.linspace(0, im.shape[1], m+1).astype(int)
    for num_x in range(len(spacing_x)-1):
        for num_y in range(len(spacing_y)-1):
            if (num_x+num_y)%2!=0:
                im[spacing_x[num_x]:spacing_x[num_x+1],spacing_y[num_y]:spacing_y[num_y+1]]=invertImg(im[spacing_x[num_x]:spacing_x[num_x+1],spacing_y[num_y]:spacing_y[num_y+1]])
        
    return im

def doCheckboardTest():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L')) #Black and white
        result_im = checkboardIm(im, 5, 3)
        vpu.showImgsPlusHists(im, result_im, title='NegativeCheckboard')


bSaveResultImgs = True
def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile))#.convert('L'))  # from Image to array #'RGB' to have all 3 color chanels 'L' for Greyscale
        for test in tests:
            out = eval(test)(im)
            im2 = out[0].astype(int) #changing the values to int for the visualization
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1],"cumulative histogram" if test=="testHistEq" else None)
            if bSaveResultImgs:
                saveImage(imfile, im2, test)

if __name__== "__main__":
    #doTests()
    doCheckboardTest()

