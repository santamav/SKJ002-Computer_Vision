from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import glob
import sys

from skimage import feature
from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough_line

from scipy import ndimage as ndi
from copy import deepcopy
import math

sys.path.append("Lab1/")
import visualPercepUtils as vpu

bLecturerVersion=False
# try:
#     import p4e
#     bLecturerVersion=True
# except ImportError:
#     pass # file only available to lecturers

#Apply the sobel filter with through a convolution
def convolveSobel(im, axis=1):
    if(axis == 0):
        mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        
    return filters.convolve(im, mask)

def testSobel(im, params=None):
    # gx = filters.sobel(im, 1)
    threshold = params['threshold']
    # Normalize image values
    im = im / 255
    # Compute the convolutions
    gx_convolved = convolveSobel(im, 1)
    gy_convolved = convolveSobel(im, 0)
    # Get the magnitude
    magnitude = np.sqrt(gx_convolved**2 + gy_convolved**2)
    # Binarize the image to 0 and 1
    magnitude = (magnitude > threshold).astype(float)
    return [magnitude]

def testCanny(im, params=None):
    sigma = params['sigma']
    edge = feature.canny(im, sigma=sigma, low_threshold=0.2 * 255, high_threshold=0.25 * 255, use_quantiles=False)
    return [edge]


def testHough(im, params=None):
    edges = testCanny(im, params)[0]
    numThetas = 200
    H, thetas, rhos = hough_line(edges, np.linspace(-np.pi/2, np.pi/2, numThetas))
    print("# angles:", len(thetas))
    print("# distances:", len(rhos))
    print("rho[...]",rhos[:5],rhos[-5:])
    return [np.log(H+1), (H, thetas, rhos)] # log of Hough space for display purpose


def findPeaks(H, thetas, rhos, nPeaksMax=None):
    if nPeaksMax is None:
        nPeaksMax = np.inf
    return hough_line_peaks(H, thetas, rhos, num_peaks=nPeaksMax, threshold=0.15 * np.max(H), min_angle=20, min_distance=15)


# -----------------
# Test image files
# -----------------
path_input = './Lab4/imgs-P4/'
path_output = './Lab4/imgs-out-P4/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'cuadros.png']  # cuadros, lena

# --------------------
# Tests to perform
# --------------------
bAllTests = False
if bAllTests:
    tests = ['testSobel', 'testCanny', 'testHough']
else:
    tests = ['testHough']
    #tests = ['testCanny']
    #tests = ['testHough']
    #tests = ['testCannyForValues']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testSobel': 'Detector de Sobel',
             'testCanny': 'Detector de Canny',
             'testHough': 'Transformada de Hough'}

bAddNoise = True
bRotate = True

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

    return im2

def testCannyForValues(im, params=None):
    sigma = [1, 1, 3, 3]
    T1 = [0.1, 0.4, 0.1, 0.01]
    T2 = [0.2, 0.6, 0.2, 0.02]
    
    # add salt and peper noise
    dirty = addSPNoise(im, 3)
    # gaussian filter to the image
    dirty = filters.gaussian_filter(dirty, 1)
    
    result = []
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(dirty, cmap='gray')
    ax[0].set_title("input")
    for i in range(len(sigma)):
        edge = feature.canny(dirty, sigma=sigma[i], low_threshold=T1[i] * 255, high_threshold=T2[i] * 255, use_quantiles=False)
        ax[i+1].imshow(edge, cmap='gray')
        ax[i+1].set_title(f"Sigma: {sigma[i]}, T1: {T1[i]}, T2: {T2[i]}")
        result = edge
        
    plt.show()
    return [result]
    
rotation = 45

def doHoughWithRotations():
    global rotation  # Add this line to access the global variable
    rotations = [0, 15, 45, 90, 180]
    
    for r in rotations:
        rotation = r
        doTests()
        
# -----------------------
# Exercise 4
# -----------------------
def HOG(im, nbins):
    # Compute the gradients
    gx = filters.sobel(im, 1)
    gy = filters.sobel(im, 0)
    # Compute the magnitude and the angle
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    # Create the histogram
    hog_descriptor = np.zeros(nbins)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            bin = int(angle[i, j] / 20)
            hog_descriptor[bin] += magnitude[i, j]
    return hog_descriptor
    


def doTestHOG():
    nbins = 9
    for file in files:
        # Load the image
        im = np.array(Image.open(file).convert('L'))
        # Compute the HOG
        hog_descriptor = HOG(im, nbins)
        plt.bar(range(nbins), hog_descriptor, width=0.5, edgecolor='black')
        plt.xticks(range(nbins), labels=range(0, 180, 20))
        plt.xlabel('Orientation bins')
        plt.ylabel('Magnitude')
        plt.title('Histogram of Oriented Gradients')
        plt.show()


def doTests():
    print("Testing on", files)
    nFiles = len(files)
    nFig = None
    for test in tests:
        if test in ["testSobel", "testCannyForValues"]:
            params = {}
            params['threshold'] = 0.5
        elif test in ["testCanny", "testHough"]:
            params = {}
            params['sigma'] = 5  # 15
        if test is "testHough":
            pass  # params={}

        for i, imfile in enumerate(files):
            print("testing", test, "on", imfile)

            im_pil = Image.open(imfile).convert('L')
            im = np.array(im_pil)  # from Image to array

            if bRotate:
                im = ndi.rotate(im, rotation, mode='nearest')

            if bAddNoise:
                im = im + np.random.normal(loc=0, scale=5, size=im.shape)

            outs_np = eval(test)(im, params)
            print("num ouputs", len(outs_np))
            if test is "testHough":
                outs_np_plot = outs_np[0:1]
            else:
                outs_np_plot = outs_np
            nFig = vpu.showInFigs([im] + outs_np_plot, title=nameTests[test], nFig=nFig, bDisplay=True)  # bDisplay=True for displaying *now* and waiting for user to close

            if test is "testHough":
                H, thetas, rhos = outs_np[1]  # second output is not directly displayable
                peaks_values, peaks_thetas, peaks_rhos = findPeaks(H, thetas, rhos, nPeaksMax=None)
                vpu.displayHoughPeaks(H, peaks_values, peaks_thetas, peaks_rhos, thetas, rhos)
                if bLecturerVersion:
                    p4e.displayLines(im, peaks_thetas, peaks_rhos, peaks_values) # exercise
                    plt.show(block=True)
                # displayLineSegments(...) # optional exercise

    plt.show(block=True)  # show pending plots (useful if we used bDisplay=False in vpu.showInFigs())


if __name__ == "__main__":
    #doTests()
    #doHoughWithRotations()
    doTestHOG()
