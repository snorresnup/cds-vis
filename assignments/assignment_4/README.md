# Assignment 4:
### Short description:
This project shows how you can use the pretrained CNN model FaceNet to detect human faces in image data. The script lists the number of faces detected in each image of three different Swiss newspapers across 23 decades and saves the data to a .csv file, that also displays the percentage of images containing faces for each decade and for each newspaper. The script plots the percentage of faces in each newspaper across the decades. Furthermore the project contains a notebook that tests the model on a single image, and displays where the model detects a face, showcasing the accuracy of the model. 

### Data source:
The dataset consists of a corpus of historic Swiss newspaper articles: : the Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). The data can be found [here](https://zenodo.org/records/3706863).

### Repository structure:

- `in` folder in which to manually put downloaded data.
- `out` folder containing the `faces.csv`, `grouped_faces.csv`, `model_test.png` and `percentage_faces.png` files.
- `src` folder containg the `script.py` and `model_test.ipynb` files.
- `README.md` file.
- `requirements.txt` file containing all required packages.
- `run.sh` file to run the script.
- `setup.sh` file to create virtual environment and install required packeges prior to running the script.

### Reproducing the analysis:
1. Download the date from [here](https://zenodo.org/records/3706863), open the zipfile and place the `images` folder in the `in` folder.

2. In order to do the setup before running the script, create a virtual environment, install the required packages and deactivate the virtual environment again by running `bash setup.sh` in the terminal.

3. In order to run the script, activate the virtual environment, run the script and deactivate the virtual environment by running `bash run.sh` in the terminal.

4. In order to see the results of the script, open the `out` folder.

### Discussion/summary:
The results of the project show that the amount of faces in newspaper articles increase over time. There are difficulties with the script, because some of the images seems to be too big to be opened, but this is only the case with 6 out of 4624 images, so their absence from the final result is not crucial. 

<img src="out/model_test.png" width="425">

The `model_test.png` image saved to the `out` folder, shows how the model only detects one face on an image that contains at least 6 faces easily detected by the human eye, which undermines the accuracy of the model, creating uncertainty as to the realiabilty of the final results. In the light of this, it is woth to consider, that the amount of faces in newspapers might not increase through the decades, but instead it is the simply the case that the faces in the newspaper articles become more easily recognizable for the model?

### Limitations and improvement
- In order to achieve higher accuracy, one could do a more thorough preprocessing in order to reduce noise and enhance existing features.
- The dataset is very big and the CNN is computionally intensive and takes a very long time to run.