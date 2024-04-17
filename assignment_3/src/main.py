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
from sklearn.model_selection import train_test_split

# for plotting
import numpy as np
import matplotlib.pyplot as plt

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

    image_arrays = [img_to_array(image['image']) for image in images]
    image_reshaped = [image_array.reshape((image_array.shape[0], image_array.shape[1], image_array.shape[2])) for image_array in image_arrays]
    image_preprocessed = [preprocess_input(image_reshape) for image_reshape in image_reshaped]
    
    return image_preprocessed

# build model
def build_model(num_classes):
    
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, 
               activation='relu')(flat1)
    output = Dense(num_classes, 
               activation='softmax')(class1)

    # define new model
    model = Model(inputs = model.input, outputs = output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def plot_loss(H, epochs, output_dir):
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
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.show()

# Main function for running the task
def main(data_path, output_dir):

    # Load data
    images = load_images(data_path)
    image_preprocessed = preprocessing(images)

    # Labels
    labelNames = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform([image["label"] for image in images])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(image_preprocessed, labels, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model(len(labelNames))

    # Train the model
    H = model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, batch_size=128, epochs=10)

    # Plot training history
    plot_loss(H, 10, output_dir)

    # Evaluate the model
    predictions = model.predict(np.array(X_test), batch_size=128)
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=labelNames)
    print(report)

    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

# Running main()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrained image embeddings for document classification")
    parser.add_argument("--data_path", type=str, default="../../../cds-vis-data/Tobacco3482", help="Data path")
    parser.add_argument("--output_dir", type=str, default="../out", help="Directory to save results")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)