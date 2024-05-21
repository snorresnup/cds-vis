import os
import sys

import numpy as np
import cv2
from joblib import dump, load

import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

# import data
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# preprocess
def preprocessing(X_train, X_test):
    X_train_results = []
    X_test_results = []

    for image in X_train:
        greyed_X_train = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_X_train = greyed_X_train/255.0
        X_train_results.append(scaled_X_train)
    reshaped_X_train = np.array(X_train_results).reshape(-1, 1024)

    for image in X_test:
        greyed_X_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_X_test = greyed_X_test/255.0
        X_test_results.append(scaled_X_test)
    reshaped_X_test = np.array(X_test_results).reshape(-1, 1024)
    return reshaped_X_train, reshaped_X_test

# train neural network classifier
def NeuNetClassifier(reshaped_X_train, y_train, reshaped_X_test, y_test):
    clf = MLPClassifier(activation = "logistic",
                    hidden_layer_sizes = (20, ),
                    max_iter = 1000,
                    random_state = 42).fit(reshaped_X_train, y_train)

    # loss curve
    plt.plot(clf.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    # save loss curve
    plt.savefig(os.path.join("out","loss_curve.png"))

    # calculate predictions
    y_pred = clf.predict(reshaped_X_test)

    # classification metrics
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    NeuNetReport = metrics.classification_report(y_test, y_pred, target_names = labels)
    
    f = open('out/classification_report_NN.txt', 'w') # open in write mode
    f.write(NeuNetReport) # write the variable into the txt file 
    f.close() 

def main():
     (X_train, y_train), (X_test, y_test) = cifar10.load_data() 
     reshaped_X_train, reshaped_X_test = preprocessing(X_train, X_test)
     NeuNetClassifier(reshaped_X_train, y_train, reshaped_X_test, y_test)

if __name__=="__main__": #if it's executed from the command line run the function "main", otherwise do NOTHING
    main()
