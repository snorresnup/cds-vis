# load packages
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# load images
def load_images():
    data_path = os.path.join("in","Tobacco3482-jpg")
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

def preprocessing(images):

    image_arrays = [img_to_array(image['image']) for image in images]
    image_reshaped = [image_array.reshape((image_array.shape[0], image_array.shape[1], image_array.shape[2])) for image_array in image_arrays]
    image_preprocessed = [preprocess_input(image_reshape) for image_reshape in image_reshaped]
    
    return image_preprocessed

# build model
def build_model():
    
    model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add classifier layers
    flat1 = Flatten()(model.output)
    class1 = Dense(128, 
               activation='relu')(flat1)
    output = Dense(10, 
               activation='softmax')(class1)

    # define new model
    model = Model(inputs = model.input, outputs = output)

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_loss(H, epochs):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    # Save the plots
    output_dir = os.path.join("out")
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.show()

def main():

    # define paths
    data_path = os.path.join("in","Tobacco3482-jpg")
    output_dir = os.path.join("out")

    # load data
    images = load_images()
    image_preprocessed = preprocessing(images)

    # labels
    labelNames = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

    # binarize labels
    lb = LabelBinarizer()
    labels = lb.fit_transform([image["label"] for image in images])

    # split data
    X_train, X_test, y_train, y_test = train_test_split(image_preprocessed, labels, test_size=0.2, random_state=42)

    # build and compile the model
    model = build_model()

    # train model
    H = model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, batch_size=128, epochs=10)

    # plot training history
    plot_loss(H, 10)

    # evaluate
    predictions = model.predict(np.array(X_test), batch_size=128)
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=labelNames)
    print(report)

    # save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

# Running main()
if __name__ == "__main__":
    main()