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





def segment_by_colour(img, al, ah, bl, bh):
    mask = rgb2lab(img)
    img1 = np.logical_and(mask[:, :, 2] > bl, mask[:, :, 2] < bh)
    img2 = np.logical_and(mask[:, :, 1] > al, mask[:, :, 1] < ah)
    mask = np.logical_and(img1, img2)


    mask = mask.astype(int)
    return mask






def get_figure(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    shape = "Other"

    if len(approx) == 3:
        shape = "Triangle"
    return shape





if __name__ == '__main__':
    #1)Opening and blur
    image = imread("MicrosoftTeams-image.png")
    image = rgba2rgb(image)
    image_gray = rgb2gray(image)
    image_gray = gaussian(image_gray)

    #2)get mask without background
    thresh = image_gray > 0.25
    thresh = morphology.opening(thresh)
    thresh = thresh.astype(int)
    plt.imshow(thresh)
    plt.show()
    labels = label(thresh)
    plt.imshow(labels)
    plt.show()
    labels = np.where(labels == 1, 0, 1)

    plt.imshow(labels)
    plt.show()
    for i in range(17):
        labels = morphology.erosion(labels)

    plt.imshow(labels)
    plt.show()

    #3)Find and remove all what is not triangles
    contours, hierarchy = cv2.findContours(img_as_ubyte(labels.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    figure = []
    new_labels=img_as_ubyte(labels)
    image1=[]
    for i in contours:
        figure.append(get_figure(i))
    for i in range(len(figure)):
        if figure[i]!="Triangle":
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(new_labels, (x, y), (x + w, y + h), (0, 0, 0), -1)
    plt.imshow(new_labels)
    plt.show()
    labels=new_labels

    #4)use the mask on image
    gray_three = cv2.merge([labels, labels, labels])
    new_image = image * gray_three
    plt.imshow(new_image)
    plt.show()

    #5)segment white colour
    mask = segment_by_colour(new_image, 0, 20, -20, 0)
    for i in range(5):
        mask=morphology.dilation(mask)
    mask=morphology.opening(mask,disk(5))
    plt.imshow(mask)
    plt.show()

    #6)change white color
    for i in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[i][x]==1:
                image[i][x]=[0,1,1]
    plt.imshow(image)
    plt.show()
    imsave("result.png",image)





