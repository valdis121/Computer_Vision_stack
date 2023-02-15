from keras.datasets import mnist
import numpy as np
from keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
from skimage.util import random_noise
from tensorflow import keras
from keras.preprocessing import image


def noise(image):
    noise = random_noise(image, mode='s&p', amount=0.2)


    noise = np.array(255 * noise, dtype=np.uint8)
    return noise
def train(mnist_train):
    x_train = mnist_train.astype('float32') / 255.
    x_train1 = np.array([noise(x) for x in x_train])

    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)


    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    autoencoder.fit(x_train1, x_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    )
    autoencoder.save('autoencoder')
    return autoencoder
def test(path_to_eval_model,y_test,autoencoder,mnist_test):
    x_test = mnist_test.astype('float32') / 255.
    x_test1 = np.array([noise(x) for x in x_test])
    y_test = keras.utils.to_categorical(y_test, 10)
    pred = autoencoder.predict(x_test1)
    eval_model = keras.models.load_model(path_to_eval_model)

    plt.imshow(x_test[0])
    plt.show()
    plt.imshow(x_test1[0])
    plt.show()
    plt.imshow(pred[0])
    plt.show()

    score1 = eval_model.evaluate(x_test, y_test, verbose=0)
    print("Clean loss:", score1[0])
    print("Clean accuracy:", score1[1])
    score2 = eval_model.evaluate(x_test1, y_test, verbose=0)
    print("Noise loss:", score2[0])
    print("Noise accuracy:", score2[1])
    score3 = eval_model.evaluate(pred, y_test, verbose=0)
    print("Denoise loss:", score3[0])
    print("Denoise accuracy:", score3[1])
def predict(image1,model):
    image1=image.img_to_array(image1)
    image1=np.expand_dims(image1,axis=0)
    return model.predict(image1)



if __name__=="__main__":
    (mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
    enc=train(mnist_train)
    test('eval_model',y_test,enc,mnist_test)
    imag=image.load_img('C:\\Users\\38095\\OneDrive\\Pictures\\mist.png',target_size=(28,28),color_mode='grayscale')
    img=predict(imag,enc)
    plt.imshow(img.reshape(28,28))
    plt.show()

