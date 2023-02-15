import cv2
import numpy as np


image=cv2.imread("C:\\Users\\38095\\OneDrive\\Pictures\\image.jpg")

imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 136, 190], dtype="uint8")
upper = np.array([13, 255, 252], dtype="uint8")
mask = cv2.inRange(imagehsv, lower, upper)
kernel = np.ones((5,5),np.uint8)
mask= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

cv2.drawContours(image, contours, -1, (0,0,0), 3)
cv2.imshow("image",image)
cv2.waitKey(0)