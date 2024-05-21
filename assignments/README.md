# Assignment 1:
### Short description:
This assignment shows how you can make two different image search algorithms, one using colour histograms and one using a convlutional neural network. By generating colour histograms for each image in a database of different images of flowers, normalizing the histograms to a value between 0 and 1, and then calculating the average distance between colour channels, it finds the 5 images with lowest distance to a reference image. The other algorithm find the 5 most similar images by using a VG166 pretrained convolutional neural network and using K-Nearest Neighbours.

### Data source:
The dataset consists of 17 categories of flowers each with 80 images and a total of 1360. The data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

### Repository structure:
The repository contains:

- `in` folder in which to manually put downloaded data.
- `out` folder containing .csv files for each search algorithms 5 most similar images.
- `refs` folder containg the reference image and the two images with most similar colour histograms.
- `src` folder containg the `script.py` file
- `README.md` file
- `requirements.txt` file containing all required packages 
- `run.sh` file to run the script
- `setup.sh` file to create virtual environment and install required packeges prior to running the script

### Reproducing the analysis:

1. Download the date from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), open the zipfile and place the `jpg` folder in the `in` folder.

2. In order to do the setup before running the script, create a virtual environment, install the required packages and deactivate the virtual environment again by running `bash setup.sh` in the terminal.

3. In order to run the script, activate the virtual environment, run the script and deactivate the virtual environment by running `bash run.sh` in the terminal.

4. In order to see the results of the script, showing the 5 most similar images to the reference image, open the `distance.csv` file in the `out` folder.

### Discussion/summary:
The reference image is shown below, and the most and second most similar image are displayed below that, showing how the colour histogram image search algorithm in the first case finds a different flower, but with the same colour, and in the second case finds a flower only partially similarly coloured. It is therefore safe to say, that the efficiency of a colour histogram search algorithm is very limited. 

<img src="refs/image_0001.jpg" width="500">
<img src="refs/image_0928.jpg" width="500">
<img src="refs/image_0876.jpg" width="500">

The CNN search algorithm however is much more precise. The first 80 images contain the first of 17 categories of flowers, and the CNN has placed all of the 5 most similar images within the first 80 images, meaning that they are all correctly identified.

### Limitations and improvement
- The histogram approach is innacurate, because it only catches the colour distribution and does not catch the spatial distribution of the colours. 
- An improvement would be a histogram based image search algorithm, that includes ways of analysing the spatial distribution of the colours. 
- The CNN approach is accurate, but computationally intensive.


# Assignment 2:
### Short description:
This assignment shows how you can use logistic regression and a neural network in order to classify different classes of images in large dataset. 

### Data source:
The data is the CIFAR-10 dataset consisting of 60000 32x32 colour images in 10 classes, with 6000 images per class. The data can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

### Repository structure:
The repository contains:

- `out` folder containing the classification reports of each classifier and loss curve for the neural network classifier
- `src` folder containg the `NeuNet.py` and `LogReg.py` files
- `README.md` file
- `requirements.txt` file containing all required packages 
- `run.sh` file to run the script
- `setup.sh` file to create virtual environment and install required packeges prior to running the script

### Reproducing the analysis:

1. In order to do the setup before running the script, create a virtual environment, install the required packages and deactivate the virtual environment again by running `bash setup.sh` in the terminal.

2. In order to run the script, activate the virtual environment, run the script and deactivate the virtual environment by running `bash run.sh` in the terminal.

3. In order to see the results of the scripts, showing classification reports for the logistic regression classifier and the neural network classifier open the `out` folder.

### Discussion/summary:
The classification report for each classifier shows a higher performance for the neural network, with an average of 0.37, than for the logistic regression with an average of 0.31. The loss curve shows how the loss value in the training of the neural network model quickly decreases and converges around 300 iterations, thus having learned as much as possible from the data and reaching near optimal perfomance.

### Limitations and improvement
- Both of the model are perform relatively poorly, suggesting a need for a more complex model such as a convolutional neural network, or simply adjusting the parameters in the existing neural network model and performing more augmentation on the training data, such as colour adjustments, flipping and rotating the images. 


# Assignment 3:
### Short description:
This assignment shows how you can train a VGG16 convolutional neural network to do document classification determining the typeof the document. The script trains the VGG16 model on the `Tobacco3482` dataset and ouputs the learning curves and the classification report for the model.

### Data source:
The `Tobacco3482` dataset contains 3482 files of 10 different document types. The data can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg/data).

### Repository structure:

- `in` folder in which to manually put downloaded data.
- `out` folder containing a classification report and learnings curves
- `src` folder containg the `main.py` file
- `README.md` file
- `requirements.txt` file containing all required packages 
- `run.sh` file to run the script
- `setup.sh` file to create virtual environment and install required packeges prior to running the script

### Reproducing the analysis:
1. Download the date from [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg/data), open the zipfile and place the `Tobacco3482-jpg` folder in the `in` folder.

2. In order to do the setup before running the script, create a virtual environment, install the required packages and deactivate the virtual environment again by running `bash setup.sh` in the terminal.

3. In order to run the script, activate the virtual environment, run the script and deactivate the virtual environment by running `bash run.sh` in the terminal.

4. In order to see the results of the script, open the `classification_report.txt` and the `learning_curves.png` files in the `out` folder.

### Discussion/summary:
The classification report shows a generally high precision with a weighted average f1-score of 0.71, but the variance of precision between classes is quite high, the highest being 0.93 and the lowest being 0.45.
The learning curves shows the loss curve decreases over epochs while the accuracy curve increases, indicating that the model is effectively improving. However, the validation loss and accuracy does not perform as well, indicating that the model is overfitting, since it perfoms well on the training data, but not on the unseen validation data.

![Reference Image](out/learning_curves.png)

### Limitations and improvement
- The classification report shows a high performance on the classes with larger support, such as `Email` and `Letter`, and a lower performance on classes with less support, such as `Report` and `Note`, indicating that the model needs more data for these classes in order to perform better.
- The model seems to be overfitting, and could be improved by adjusting the model, for example by addig dropout regularization. 


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