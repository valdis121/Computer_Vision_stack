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


def color_segment(img, al, ah, bl, bh):
    mask = rgb2lab(img)
    img1 = np.logical_and(mask[:, :, 2] > bl, mask[:, :, 2] < bh)
    img2 = np.logical_and(mask[:, :, 1] > al, mask[:, :, 1] < ah)
    mask = np.logical_and(img1, img2)
    mask = mask.astype(int)
    return mask

def count_percent_of_colour(img,circle_pixels,color):
        if color in ["orange","Orange"]:
            img=color_segment(img,20,100,50,100)
            plt.imshow(img)
            plt.show()
            pixels=0
            for i in range(len(img)):
                for x in range(len(img[i])):
                    if img[i][x]==1:
                        pixels+=1
            return (pixels/circle_pixels)*100
        elif color in ["red","Red"]:
            img=color_segment(img,60,100,20,50)
            plt.imshow(img)
            plt.show()
            pixels=0
            for i in range(len(img)):
                for x in range(len(img[i])):
                    if img[i][x]==1:
                        pixels+=1
            return (pixels/circle_pixels)*100
        elif color in ["blue","Blue"]:
            img=color_segment(img,-100,100,-100,0)
            plt.imshow(img)
            plt.show()
            pixels=0
            for i in range(len(img)):
                for x in range(len(img[i])):
                    if img[i][x]==1:
                        pixels+=1
            return (pixels/circle_pixels)*100
        elif color in ["green","Green"]:
            img=color_segment(img,-100,0,0,100)
            plt.imshow(img)
            plt.show()
            pixels=0
            for i in range(len(img)):
                for x in range(len(img[i])):
                    if img[i][x]==1:
                        pixels+=1
            return (pixels/circle_pixels)*100
if __name__ == '__main__':
    image = imread("MicrosoftTeams-image (1).png")
    image = rgba2rgb(image)
    image_gray = rgb2gray(image)
    thresh = image_gray < 0.9
    plt.imshow(thresh)
    plt.show()
    labels = label(thresh)
    labels = np.where(labels == 1, 1, 0)
    circle_pixels=0
    for i in range(len(labels)):
        for y in range(len(labels[i])):
            if labels[i][y]==1:
                circle_pixels+=1
    gray_three = cv2.merge([labels, labels, labels])
    new_image=gray_three*image
    plt.imshow(new_image)
    plt.show()

    print(str(count_percent_of_colour(new_image,circle_pixels,"orange"))+"%")

