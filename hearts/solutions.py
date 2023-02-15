import cv2
from skimage.morphology import label
import numpy as np
from skimage.morphology import disk
import matplotlib.pyplot as plt
from skimage.color import rgb2lab ,rgb2gray, rgba2rgb
from skimage import morphology
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.filters import gaussian
from skimage.filters import try_all_threshold
from skimage import exposure

def segment_by_colour(img, al, ah, bl, bh):
    mask = rgb2lab(img)
    img1 = np.logical_and(mask[:, :, 2] > bl, mask[:, :, 2] < bh)
    img2 = np.logical_and(mask[:, :, 1] > al, mask[:, :, 1] < ah)
    mask = np.logical_and(img1, img2)


    mask = mask.astype(int)
    return mask

def image1():
    image = imread("hearts 1.png")
    image_gray = rgb2gray(image)
    return image_gray > 0.07
def image2():
    image = imread("hearts 2.png")
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.2)
    plt.hist(img_adapteq.ravel(), bins=256)
    plt.show()
    return img_adapteq>0.42
def image3():
    image = imread("hearts 3.png")
    mask1 = segment_by_colour(image,0,100,-100,-40)
    return mask1
if __name__=="__main__":
    plt.imshow(image3())
    plt.show()


