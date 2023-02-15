import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

def get_emotion(model,img):
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    test_img= cv2.resize(img, (48,48), interpolation = cv2.INTER_AREA)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = test_img.reshape(1, 48, 48, 1)
    result = model.predict(test_img)
    result = list(result[0])

    img_index = result.index(max(result))
    return label_dict[img_index]
img = cv2.imread('5a71ad9d56892c639ab952c00213be05.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Для детектирования лиц используем каскады Хаара
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
bboxes = classifier.detectMultiScale(gray_img)
faces=[]
for i in bboxes:
    x, y, w, h = i
    img_new=gray_img[y:y+h, x:x+w]
    faces.append(img_new)
    cv2.imshow('image', img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model=get_model()
model.load_weights('model1.h5')

for i in faces:
    print(get_emotion(model,img))

for box in bboxes:
    # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
