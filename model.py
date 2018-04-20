import csv
# import math
import cv2
from enum import Enum
# import types
# import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
# from keras import Dropout
# from keras.layers import MaxPooling2D
# from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

from keras import backend as K

# consider these as input parameters
IMG_PATH = './data/IMG/'
CSV_FILE = './data/driving_log.csv'


class MyGeneratorType(Enum):
    TRAIN = 1
    VALIDATION = 2


def resize_batch_image(images):
    import tensorflow as tf
    resized = tf.image.resize_images(images, (66, 200))
    return resized


def build_model(row, col, ch):  # similar to nvidia model
    model = Sequential()
    # cropping the upper (sky) and bottom (hood of car) part of images to focus on the road
    # image cropping outside the model on the CPU is relatively slow.
    model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(row, col, ch)))
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(resize_batch_image))
    # Normalize images
    model.add(Lambda(lambda x: x / 127.5 - 1.))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))

    print(model.summary())

    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    # model.compile(loss='mse', optimizer='adam')
    return model


def read_csv_file_for_data(csvfile):
    samples = []
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            samples.append(row)
    return samples


def flip_image_horizontally(image):
    return cv2.flip(image, 1)


def change_brighteness_of_image(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = 0.25 + np.random.standard_normal()  # some random brightness + bias to avoid dark image
    np.multiply(img_hsv[:, :, 2], brightness, out=img_hsv[:, :, 2], casting="unsafe")  # for V channel

    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def rotate_image(image):
    num_rows, num_cols = image.shape[:2]
    angle = np.random.standard_normal() * 20 + 1
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return angle, cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))


# def shift_image(image):


def augment_training_data(car_images, steering_angles, batch_sample):
    img_choice_1 = np.random.randint(3) + (-3)
    flipped_img = flip_image_horizontally(car_images[img_choice_1])
    car_images.append(flipped_img)
    steering_angles.append(steering_angles[img_choice_1] * (-1.0))

    img_choice_2 = np.random.randint(3) + (-3)
    lighter_img = change_brighteness_of_image(car_images[img_choice_2])
    car_images.append(lighter_img)
    steering_angles.append(steering_angles[img_choice_2])

    val, rotated_img = rotate_image(car_images[-3])  # only center image
    car_images.append(rotated_img)
    steering_angles.append(steering_angles[-3] - float(val) / 100.)


def get_images_for_batch_sample(car_images, batch_sample):
    # read in images from center, left and right cameras
    for i in range(3):
        source_path = batch_sample[i]
        filename = source_path.split('/')[-1]
        current_path = IMG_PATH + filename
        img_bgr = cv2.imread(current_path)
        # Keep in mind that training images are loaded in BGR color space using cv2
        # while drive.py load images in RGB to predict the steering angles.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # print(img_rgb.shape)
        car_images.append(img_rgb)


def get_steering_angles_for_batch_sample(steering_angles, batch_sample):
    steering_center = float(batch_sample[3])
    # use correction for left and right images
    # create adjusted steering measurements for the side camera images
    correction = 0.2  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    steering_angles.extend([steering_center, steering_left, steering_right])


def img_generator(samples, flag, batch_size=32):
    n_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            # batch_samples = samples[offset: n_samples] \
            #     if (offset + batch_size > n_samples) \
            #     else samples[offset: offset + batch_size]
            batch_samples = samples[offset: offset + batch_size]
            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                get_images_for_batch_sample(car_images, batch_sample)
                get_steering_angles_for_batch_sample(steering_angles, batch_sample)
                if flag == MyGeneratorType.TRAIN:
                    augment_training_data(car_images, steering_angles, batch_sample)
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            assert len(X_train) == len(y_train), \
                "len(X_train) = {}, len(y_train) = {}".format(len(X_train), len(y_train))

            yield shuffle(X_train, y_train, random_state=7)


# def plot_loss(history):
#     # print the keys contained in the history object
#     print(history.history.keys())

#     # plot the training and validation loss for each epoch
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model mean squared error loss')
#     plt.ylabel('mean squared error loss')
#     plt.xlabel('epoch')
#     plt.legend(['training set', 'validation set'], loc='upper right')
#     plt.show()


def pipeline():
    samples = read_csv_file_for_data(CSV_FILE)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # compile and train the model using the generator function
    # these two should be different
    BATCH = 16
    train_generator = img_generator(train_samples, MyGeneratorType.TRAIN, batch_size=BATCH)
    validation_generator = img_generator(validation_samples, MyGeneratorType.VALIDATION, batch_size=BATCH)

    row, col, ch = 160, 320, 3  # camera format
    model = build_model(row, col, ch)

    # fits the model on batches with real-time data augmentation
    # history = model.fit_generator(train_generator,

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_samples),
                        epochs=1,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples),
                        verbose=1,
                        callbacks=[early_stop]
                        )

    # save model
    model.save('mymodel.h5')

    # plot_loss(history)


if __name__ == "__main__":
    pipeline()
    K.clear_session()
