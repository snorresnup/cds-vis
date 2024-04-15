# load packages
import os
import argparse

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt



# list subfolders
def list_subfolders(directory):
    subfolders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    return subfolders





# load images
def load_images(data_path):
    dirs = sorted(os.listdir(data_path))
    images = []

    for directory in dirs:
        subfolder = os.path.join(data_path, directory)
        image_files = sorted(os.listdir(subfolder))

        for image in image_files:
            file_path = os.path.join(subfolder, image)
            if file_path.endswith('.jpg'):
                image_data = load_img(file_path, target_size=(224, 224))
                images.append({"label": directory, "image": image_data})
                       
    return images

# preprocess
def preprocessing(images):

    for image in images:
        image_array = img_to_array(image)
        image_reshaped = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        image_preprocessed = preprocess_input(image_reshaped)
    
    return image_preprocessed

# build model?!?!?
def build_model(num_classes):
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
               activation='relu')(bn)
    dropout = Dropout(0.5)(class1)
    output = Dense(num_classes, 
               activation='softmax')(dropout)

    # define new model
    model = Model(inputs = base_model.input, outputs = output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # summarize
    model.summary()
