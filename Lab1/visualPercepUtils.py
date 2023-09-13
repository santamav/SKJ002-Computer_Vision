import numpy as np
import math
import matplotlib.pyplot as plt

def computeRowsCols(N, m, n):
    print(N, m, n)
    if m is None:
        m = math.sqrt(N)
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    else:
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    m, n = max(1, m), max(1, n)
    m, n = math.ceil(m), math.ceil(n)
    #print(m, n)
    return m, n

def showInFigs(imgs,title, nFig=None,bDisplay=False):
# open all images in separate figures without user interation
    i = 0 if nFig is None else nFig+1
    for im in imgs:
        #print(i)
        plt.figure(i)
        i+=1
        plt.imshow(im,cmap='gray',interpolation=None)#, aspect=1/1.5)#, vmin=0,vmax=255)
        plt.title(title)
    if bDisplay:
        plt.show(block=True)
    return i

def showInGrid(imgs, m=None, n=None, title="",subtitles=None):
    N = len(imgs)
    #plt.subplots_adjust(top=5) # añadido (10/10/19) para ejecutar una entrega de un estudiante
    m, n = computeRowsCols(N, m, n)
    #print(m,n)
    fig = plt.figure(figsize=(m, n))
    plt.gray()
    for i in range(1, N + 1): 
        ax=fig.add_subplot(m, n, i)
        if len(imgs[i - 1].shape) >= 2:
            plt.imshow(imgs[i - 1])
        else:
            plt.plot(imgs[i - 1])
        if subtitles is not None:
            ax.set_title(subtitles[i-1])

    fig.suptitle(title)
    #plt.savefig(title+".png")#,bbox_inches='tight')
    plt.show(block=True)

def histImg(im):
    return np.histogram(im.flatten(), 256)[0]

def showPlusInfo(data,title=None):
    plt.plot(data)
    if title is not None:
        plt.title(title)
    plt.show(block=True)

def showImgsPlusHists(im, im2, title=""):
    hists = [histImg(im), histImg(im2)]
    #print(im2.shape, hists[0].shape)
    showInGrid([im, im2] + hists, title=title)
    # alternative possibilities:
    # showInGrid(imgs)
    # showInGrid(hists)
    # showInGrid([im, im2] + hists)
    # showInGrid((imgs[0],hists[0],imgs[1],hists[1]))


# showInGrid(imgs + hists)
# showInGrid((imgs[0],hists[0],imgs[1],hists[1]))
# plt.plot(cdf)
# plt.show()


def pil2np(in_pil):
    imgs=[]
    for im_pil in in_pil:
        print(im_pil.size)
        imgs.extend([np.array(im_pil)])

    return imgs


def displayHoughPeaks(h,peaks,angles,dists,theta,rho):
    nThetas=len(theta)
    rangeThetas=theta[-1]-theta[0]
    slope=nThetas/rangeThetas
    plt.axis('off')
    plt.imshow(h, cmap='jet')
    for peak, angle, dist in zip(peaks,angles,dists):
        print("peak",peak, "at angle",np.rad2deg(angle),"and distance ", dist)
        #y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        #y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        #ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        plt.plot(slope*(angle-theta[0])+1, dist-rho[0], 'rs', markersize=0.1*peak) # size proportional to peak value
    plt.show(block=True)

def showImWithColorMap(im,cmap='spectral'):
    plt.imshow(im,cmap=cmap)
