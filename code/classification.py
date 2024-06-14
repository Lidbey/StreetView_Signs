import tensorflow as tf
from keras.layers import Flatten, Dropout, Dense
from keras import Model
from keras_preprocessing.image import ImageDataGenerator
from data import num_classes, classification_size
import numpy as np
import cv2


def get_model(name=None):
    resnet = tf.keras.applications.ResNet50V2(input_shape=(*classification_size, 3), include_top=False)
    x = resnet.output
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=resnet.input, outputs=output)
    if name:
        model.load_weights(f'{name}')
    return model


def get_generator():
    return ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.2,
                              horizontal_flip=False,
                              preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)


def get_pre_image(name):
    img = tf.keras.preprocessing.image.load_img(name)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    return tf.keras.applications.resnet_v2.preprocess_input(img_tensor)


def cut_by_bbox(img, boxes):
    boxes = np.round(boxes).astype(int)
    l = np.zeros((len(boxes), *classification_size, 3))
    for index, i in enumerate(boxes):
        box = img[i[1]:i[3], i[0]:i[2], :]
        box = cv2.resize(box, (64, 64))
        box = np.expand_dims(box, axis=0)
        l[index,] = box
    return l
