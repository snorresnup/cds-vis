import os
# adding python path
import sys
sys.path.append("..")
# openCV
import cv2
# plotting 
import matplotlib.pyplot as plt
import pandas as pd

def gen_hists(image):
    # load element
    ref_img = cv2.imread(image)
    # generate histogram
    hist = cv2.calcHist([ref_img], [0, 1, 2], None, [255, 255, 255], [0,256, 0,256, 0,256])
    # normalize histogram
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return(norm_hist)

def compare_histograms():
    filepath_ref = os.path.join("..","..","..","..", "cds-vis-data", "flowers", "image_0001.jpg")
    filepath = os.path.join("..","..","..","..", "cds-vis-data", "flowers")

    hist_ref = gen_hists(filepath_ref)

    all = sorted(os.listdir(filepath))

    results = []

    for img in all:
        input_path = os.path.join(filepath, img)
        hist = gen_hists(input_path)
        distance = round(cv2.compareHist(hist_ref, hist, cv2.HISTCMP_CHISQR), 2)
        results.append((img, distance))
        df = pd.DataFrame(results, columns=["Filename", "Distance"]).sort_values(by=["Distance"])
    
    df_5 = df.head(6)
    output_path = os.path.join("..","out")
    df_5.to_csv(os.path.join(output_path, "distance.csv"), index = False)

def main():
    compare_histograms()

if __name__=="__main__":
    main()