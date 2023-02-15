import cv2
from skimage.morphology import label
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy
import matplotlib.pyplot as plt
from skimage.color import rgb2lab ,rgb2hsv, rgb2gray, rgba2rgb
from skimage import morphology,io
import scipy.ndimage as nd
from skimage import exposure
from skimage.io import imread
from skimage import img_as_ubyte
def get_region_by_color(img, al, ah, bl, bh):

    mask = rgb2lab(img)

    img1 = np.logical_and(mask[:, :, 2] > bl, mask[:, :, 2] < bh)
    img2 = np.logical_and(mask[:, :, 1] > al, mask[:, :, 1] < ah)


    mask = np.logical_and(img1, img2)
    plt.imshow(mask)
    plt.show()
    mask = morphology.opening(mask, morphology.disk(5))

    mask = nd.binary_fill_holes(mask)
    mask = mask.astype(int)
#    print(mask)
#    mask=cv2.merge([mask, mask, mask])
#   img=mask*img
#    return img
    return mask

def get_region_by_colorHsv(img, lower,upper):
    img = rgb2hsv(img)

    mask = cv2.inRange(img, lower, upper)

    mask = morphology.opening(mask, morphology.disk(5))



    mask = nd.binary_fill_holes(mask)

    return img

def get_figure_name(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    shape_type = "Other"
    corners = len(approx)

    if corners == 5:
        shape_type = "Triangle"
    if corners == 4:
        shape_type = "Rectangle"
    if corners >= 10:
        shape_type = "Circle"
    return shape_type
image=imread("teams.png")
image=rgba2rgb(image)
image_gray=rgb2gray(image)



thresh=image_gray>0.07


thresh=thresh.astype(int)
labels=label(thresh)


labels=np.where(labels==1,0, labels)
labels=np.where(labels!=0,1, labels)

gray_three = cv2.merge([labels,labels,labels])
new_image=image*gray_three

new_image=get_region_by_color(new_image,10,127,-127,-10)
contours, hierarchy = cv2.findContours(img_as_ubyte(new_image.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

blue_figure=[]

for i in contours:
    blue_figure.append(get_figure_name(i))
print(blue_figure)





