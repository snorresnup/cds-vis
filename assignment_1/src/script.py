# Importing libraries
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Histogram search algorithm
def gen_hists(image):
    # load element
    ref_img = cv2.imread(image)
    # generate histogram
    hist = cv2.calcHist([ref_img], [0, 1, 2], None, [255, 255, 255], [0,256, 0,256, 0,256])
    # normalize histogram
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return(norm_hist)

def compare_histograms():
    filepath_ref = os.path.join("data", "jpg", "image_0001.jpg")
    filepath = os.path.join("data", "jpg")

    hist_ref = gen_hists(filepath_ref)
    
    images = [os.path.join(filepath, image) for image in os.listdir(filepath) if image.endswith('.jpg')]

    results = []

    for img in images:
        hist = gen_hists(img)
        distance = round(cv2.compareHist(hist_ref, hist, cv2.HISTCMP_CHISQR), 2)
        results.append((img, distance))
        df = pd.DataFrame(results, columns=["Filename", "Distance"]).sort_values(by=["Distance"])
    
    df_5 = df.head(6)
    output_path = os.path.join("out")
    df_5.to_csv(os.path.join(output_path, "histogram_distance.csv"), index = False)
    

# CNN VGG16 search algorithm

def load_model():
    model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))
    
    return model

def extract_features(model):
    filepath = os.path.join("data", "jpg")
    images = [os.path.join(filepath, image) for image in os.listdir(filepath) if image.endswith('.jpg')]
    input_shape = (224, 224, 3)
    feature_list = []

    filepath_ref = os.path.join("data", "jpg", "image_0001.jpg")
    ref = model.predict(preprocess_input(np.expand_dims(img_to_array(load_img(filepath_ref, target_size=(input_shape))), axis = 0)), verbose = False)
    ref_features = ref.flatten()

    for image in images:
        img = load_img(image, target_size=(input_shape))
        img_array = img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img, verbose=False)
        flattened_features = features.flatten()
        feature_list.append(flattened_features)

    return feature_list, ref_features

def find_nearest_neighbours(feature_list, ref_features):
    filepath = os.path.join("data", "jpg")
    images = [os.path.join(filepath, image) for image in os.listdir(filepath) if image.endswith('.jpg')]
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine').fit(feature_list)
    distances, indices = neighbors.kneighbors([ref_features])
    df_5 = pd.DataFrame({
            "Filename": [images[i] for i in indices[0][0:]],
            "Distance": [float(f"{dist:.2f}") for dist in distances[0][0:]]})
    output_path = os.path.join("out")
    df_5.to_csv(os.path.join(output_path, "cnn_distance.csv"), index = False)


def main():
    compare_histograms()
    model = load_model()
    feature_list, ref_features = extract_features(model)
    find_nearest_neighbours(feature_list, ref_features)


if __name__=="__main__":
    main()