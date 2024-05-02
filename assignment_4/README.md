# Assignment 4:

### Short description:
This project shows how you can use the pretrained CNN model FaceNet to detect human faces in image data. The script lists the number of faces detected in each image of three different Swiss newspapers across 23 decades and saves the data to a .csv file, that also displays the percentage of images containing faces for each decade and for each newspaper. The script plots the percentage of faces in each newspaper across the decades. Furthermore the project contains a notebook that tests the model on a single image, and displays where the model detects a face, showcasing the accuracy of the model. 

### Data source:
The dataset consists of a corpus of historic Swiss newspaper articles: : the Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). The dataset is located in the `cds-vis-data` folder in the folder `newspapers`.

### Repository structure:

`face_detection()`
This functions loads the pretrained CNN model for face detection.

`process()`
This function loads the images, executes the face_detection function on each image, and saves the number of faces detected for each file to the `faces.csv` file.

`save_csv(`
This functions groups the detected faces by decade, and calculates the percentage of newsarticles with detected faces for each decade and newspaper. 

`plot_percentage()`
This functions plots the percentage of newsarticles with faces progressively through the decades for each newspaper.

`main()`
This function saves the data_path and the output_path and runs the `process()`, `save_csv()` and `plot_percentage()` functions.

### Reproducing the analysis:
pip install -r requirements.txt
python script.py

### Discussion/summary:
The results of the project show that the amount of faces in newspaper articles increase over time. There are difficulties with the script, because some of the images seems to be too big to be opened, but this is only the case with 6 out of 4624 images, so their absence from the final result is not crucial. The `model_test.png` image saved to the `out` folder, shows how the model only detects one face on an image that contains at least 6 faces easily detected by the human eye, which undermines the accuracy of the model, creating uncertainty as to the realiabilty of the final results. In the light of this, it is woth to consider, that the amount of faces in newspapers might not increase through the decades, but instead it is the simply the case that the faces in the newspaper articles become more easily recognizable for the model?

