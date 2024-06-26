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
    return (im[::,] ** float(p) / (255 ** (p-1))   ) # try without the float conversion and see what happens

def brightenImg(im,p=2):
    return np.power(255.0 ** (p - 1) * im[::,], 1. / p)  # notice this NumPy function is different to the scalar math.pnp

def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

path_input = './Lab1/imgs-P1/'
path_output = './Lab1/imgs-out-P1/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*") #Changed from ~~files = glob.blob(path_input + "*.pgm")~~ so it reads all the files
else:
    files = [path_input + 'peppers.ppm'] # iglesia,huesos, path_input + 'iglesia.pgm'

bAllTests = False
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg', 'testCheckboard']
else:
    tests = ['testDarkenImg']#['testHistEq']#['testBrightenImg']
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image', 'testCheckboard':'Checkboard'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
               'testCheckboard':'_cb'}

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

def old_checkboardIm(im, n, m):
    im_result = im.copy()
    spacing_y = np.linspace(0, im.shape[0], n+1).astype(int)
    spacing_x = np.linspace(0, im.shape[1], m+1).astype(int)
    for num_y in range(len(spacing_y)-1):
        for num_x in range(len(spacing_x)-1):
            if (num_y+num_x)%2!=0:
                im_result[spacing_y[num_y]:spacing_y[num_y+1],spacing_x[num_x]:spacing_x[num_x+1]]=invertImg(im[spacing_y[num_y]:spacing_y[num_y+1],spacing_x[num_x]:spacing_x[num_x+1]])
        
    return im_result

def checkboardIm(im, n, m):
    height, width = im.shape[:2]
    cell_height = np.ceil(height / n).astype(int) # height//n
    cell_width = np.ceil(width / m).astype(int) # width//m
    # Create row and column indices for the cells
    row_indices = np.arange(height) // cell_height
    col_indices = np.arange(width) // cell_width
    # We can also reshape the two index arrays and sum them toguether row_indices = row.reshape(1,-1) + col.reshape(-1,1)
    # Create masks for the odd-indexed cells
    mask = (row_indices[:, np.newaxis] + col_indices) % 2 == 1
    # Use broadcasting to invert pixels in odd-indexed cells
    im_result = np.copy(im)
    im_result[mask] = invertImg(im_result[mask])
    return im_result

def doCheckboardTest():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L')) #Black and white
        result_im = checkboardIm(im, 5, 3)
        vpu.showInGrid([im, result_im])
        saveImage(imfile, result_im, 'testCheckboard')


"""
Recursively calculate the histogram for n subdivions
"""
def recursiveMultihist(im, n, result, nbins=256):
    #Calculate current level histogram
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    #Add the current imhist to the result
    result.append(imhist)
    #Break condition, either n = 0 or the image cannot be made smaller
    if(n <= 1): return
    #Calculate the subdivision ranges
    n_cuadrants = 4
    height, width = im.shape[:2]
    row_ranges = np.linspace(0, height, n_cuadrants+1).astype(int) #each image is subdivided in 4 cuadrants
    col_ranges = np.linspace(0, width, n_cuadrants+1).astype(int)
    #Calculate subdivision histograms recursively
    for i in range(n_cuadrants): #each iteration subdivides in 4
        img_chunk = im[row_ranges[i]:row_ranges[i+1], col_ranges[i]:col_ranges[i+1]]
        recursiveMultihist(img_chunk, n-1, result)
    
def multihist(im, n):
    result = []
    recursiveMultihist(im, n, result)
    vpu.showInGrid(result)
    return result

def doMultiHistTest():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L')) #Black and white
        result = multihist(im, 3)
        #TODO: Make the visualization and save the image

# ****************************************************************
# Gray level transformation fucntion T(l) = a*(e**(-alpha*(l**2)))
# n = subdivisions
# l0 = min value
# l1 = max value
# ****************************************************************
def expTransf(alpha, n, l0, l1, bInc=True):
    # Generate an array of equally spaced input values
    l_values = np.linspace(l0, l1, n)
    
    # Calculate the exponential transformation
    a = 255 if bInc else -255  # Determine 'a' based on bInc
    b = 0 if bInc else 255    # Determine 'b' based on bInc
    
    # Apply the transformation function
    transformed_values = a * np.exp(-alpha * l_values**2) + b
    return transformed_values

def transfImage(im, f):
    # Normalize the image pixel values to the [0, 255] range
    im_normalized = ((im - im.min()) / (im.max() - im.min())) * 255
    
    # Apply the transformation function 'f' to the normalized image
    transformed_im = f[im_normalized.astype(int)]
    
    # Clip values to ensure they are within the [0, 255] range
    transformed_im = np.clip(transformed_im, 0, 255).astype(np.uint8)
    
    return transformed_im

def doExpTransf():
    alpha = 0.01
    n = 256
    l0 = 0
    l1 = 255
    bInc = True
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L'))
        #Call the transfImage function for a different amount of functions
        transf_im = transfImage(im, expTransf(alpha, n, l0, l1, bInc))
        vpu.showInGrid(transf_im, "Transformed Img")


    transformed_values = expTransf(alpha, n, l0, l1, bInc)

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
    doTests()
    #I could have added the functions to the tests array instead of calling them here
    #doCheckboardTest()
    #doMultiHistTest()
    #doExpTransf()

