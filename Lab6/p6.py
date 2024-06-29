import numpy as np
from scipy import signal

from skimage.data import stereo_motorcycle, moon, astronaut, camera
from skimage.transform import SimilarityTransform, warp

from skimage.color import rgb2gray
import matplotlib.pylab as plt
from skimage.filters import gaussian
#from skimage.registration import optical_flow_ilk as LK
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from skimage.transform import rescale

import time

def display_image_pyr(imgs):
    N = len(imgs)
    m = int(np.sqrt(N))
    n = N // m
    if (m * n < N):
        m += 1
    fig = plt.figure(figsize=(8., 8.))

    for i, im in enumerate(imgs):
        fig.add_subplot(m, n, i + 1)
        plt.imshow(im, cmap='gray')
        plt.title("size: " + str(im.shape))

    plt.show(block=True)

def optical_flow(I1g, I2g, window_size, tau=1e-2, bDisplay=False):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    # Your code (1): kernels for y
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    # Your code (1): kernels for t
    kernel_t = np.ones((4,4)) / 16 # Mean filter with 4x4 size
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window

    I1g = gaussian(I1g, sigma=5, truncate=1/5)
    I2g = gaussian(I2g, sigma=5, truncate=1/5)

    # Implement Lucas-Kanade
    # for each image point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) - signal.convolve2d(I1g, kernel_t, boundary='symm', mode=mode)
    if bDisplay:
        for f in [fx,fy,ft]:
            plt.imshow(f,cmap='gray')
            plt.show(block=True)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    # iterate for each image window of size window_size * window_size
    M,N = I1g.shape
    for i in range(w, M-w):
        for j in range(w, N-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            # A is a vertically stacked matrix of Ix and Iy and b is a one column matrix of -It
            A = np.column_stack((Ix, Iy))
            b = -It
            AtA = np.matmul(A.T, A)
            Atb = np.matmul(A.T, b)

            # Verify that the smallest eigenvalue of AtA is greater than tau or
            # AtA has rank 2
            if np.min(np.linalg.eigvals(AtA)) > tau or np.linalg.matrix_rank(AtA) == 2:
                nu, residuals, rank, s = np.linalg.lstsq(AtA, Atb, rcond=None)
                u[i,j]=nu[0]
                v[i,j]=nu[1]

    return u,v

def display_optic_flow(I1, I2, u, v, title=""):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    # --- Sequence image sample

    ax0.imshow(I1, cmap='gray')
    ax0.set_title(r"$I_1$")
    ax0.set_axis_off()

    ax1.imshow(I2, cmap='gray')
    ax1.set_title(r"$I_2$")
    ax1.set_axis_off()

    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = I1.shape
    step = max(nl // nvec, nc // nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    norm = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v,u)
    ax2.imshow(angle)

    widths = np.linspace(0, 10, u.size)
    ax2.quiver(x, y, u_, v_, color='w', units='dots',
               angles='xy', scale_units='xy', lw=2, headaxislength=30, headlength=30, headwidth=15, edgecolors='w')#linewidth=2, edgecolors='w')
    ax2.set_title("Optical flow orientation and vector field")
    ax2.set_axis_off()
    fig.tight_layout()

    plt.suptitle(title)
    plt.show(block=True)


def get_real_sequence_image_pair(seq,t0,t1=None):

    assert seq in ['taxi','traffic','corridor'] # make sure the chosen sequence is available

    if t1 is None:
        t1 = t0 + 1 # if not provided the second frame, assume it is the following one

    valid_frame_ranges = {'taxi': [0, 40], 'traffic': [0, 48], 'corridor': [0, 9]}

    # make sure the chosen frames are available for the chosen sequence
    assert valid_frame_ranges[seq][1] >= t0 >= valid_frame_ranges[seq][0]
    assert valid_frame_ranges[seq][1] >= t1 >= valid_frame_ranges[seq][0]

    # read the corresponding frames
    I1, I2 = [Image.open("Lab6/imgs-P6/" + seq + "/" + seq + "_frame" +
                         str(t).zfill(3) + ".png").convert('L') for t in [t0, t1]]

    # we can downscale for either computational reasons or to have smaller motions
    resize_factor = 0.5  # you can experiment with different values

    I1 = I1.resize([int(resize_factor * s) for s in I1.size])
    I2 = I2.resize([int(resize_factor * s) for s in I2.size])

    I1 = np.array(I1) / 255
    I2 = np.array(I2) / 255

    return I1, I2

def get_synthetic_sequence_image_pair(seq, translation=(0,0), rotation=0, scale=1):
    assert seq in ['moon','astronaut','camera'] # make sure the chosen sequence is available
    # moon: low-contrast image of the moon
    # astronaut: color image of astronaut
    # camera: gray-level image of cameraman
    image0 = eval(seq)()
    if (len(image0.shape) == 3):
        image0 = rgb2gray(image0)
    I1 = np.array(image0)
    if (I1.dtype == "uint8"):
        I1 = I1 / 255.0

    # apply synthetic transformation
    shift_y, shift_x = (np.array(I1.shape[:2]) - 1) / 2.  # image center
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

    tf_transf = SimilarityTransform(scale= scale, rotation=np.deg2rad(rotation), translation=translation)
    I2 = warp(I1, (tf_shift + (tf_transf + tf_shift_inv)).inverse, preserve_range=True)

    return I1, I2


def display_magnitude_and_orientation(magnitude, angle, subtitle=""):
    # display the magnitude of OF
    fig, ax = plt.subplots()
    plt.hist(magnitude[magnitude>0].flatten(), 30, label="LK, seq="+seq, fc=(0, 0, 1, 0.2))
    plt.legend()
    plt.title(subtitle)
    plt.suptitle("Magnitude", ha='center')
    plt.show(block=True)

    # display the orientation of OF
    plt.imshow(angle,cmap='hsv') # cyclic maps: 'twilight', 'twilight_shifted', 'hsv'
    plt.title("Orientation")
    plt.show(block=True)


if __name__ == "__main__":

    bSyntheticTransf = False
    bCrop = False

    if not bSyntheticTransf:
        seq = 'traffic'
        seq = 'corridor'
        seq = 'taxi'
        I1, I2 = get_real_sequence_image_pair(seq,10)
    else:
        seq = 'moon'
        seq = 'astronaut'
        seq = 'camera'
        scale = 1.02 # 1=no change, >1: zoom in, <1: <zoom out (e.g.
        rotation = 0.0 # in degrees, positive or negative for clockwise or counter-clockwise, respectively
        translation = (0.0, 0.0) # (tx, ty) in pixel, positive or negative
        I1, I2 = get_synthetic_sequence_image_pair(seq, translation=translation, rotation=rotation, scale=scale)


    if bCrop: # if we want to work only on some smaller part of the input images I1, I2
        y1, y2 = 200, 400
        x1, x2 = 200, 400
        I1 = I1[y1:y2, x1:x2]
        I2 = I2[y1:y2, x1:x2]

    # display the images before computing the optic flow
    plt.title(r"$I_1, I_2$")
    plt.imshow(np.hstack((I1,I2)),cmap='gray')
    plt.show(block=True)

    # You are asked to experiment with different values for these hyperparameters
    window_size = 20
    tau = 0.001

    # Running the LK method
    u, v = np.zeros_like(I1), np.zeros_like(I1) # comment out this line with the proper call (below) when you are ready
    start_time = time.time()
    u,v = optical_flow(I1, I2, window_size=window_size, tau=tau)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    display_optic_flow(I1, I2, -u, -v, title="Optical flow")

    # Compute the magnitude and orientation of OF
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v,u)

    #subtitle = f"window_size={window_size}, translation={translation}, rotation={rotation}, scale={scale}"
    subtitle = f"window_size={window_size}, tau={tau}"
    display_magnitude_and_orientation(magnitude, angle, subtitle=subtitle)


