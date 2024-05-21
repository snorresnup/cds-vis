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