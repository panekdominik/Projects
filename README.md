# Requirements
Python version = 3.8.10
Packages versions:
tensorflow = 2.13.0
numpy = 1.22.4
pandas = 1.5.3
matplotlib = 3.3.4
keras = 2.13.1
skimage = 0.18.1
cv2 = 4.5.5

# Cell shape classification
This directory contains functions intended for classification of cell shapes (if you want to know more please go to: ...). Main script is called cell_classification.py. It can be called in terminal with appropriate arguments (described within this file or you can simply type -h after calling the script.) It uses helpers.py script to create a data contained in separate folders. Data must be sotred in form of images showing cells. Then labels are loaded (labels describing states of the cells in a given image). Later traning and testing subsets are created and the training via recurrent or/and convolutional neural network (RNN and CNN respectively) is performed. You can modify number of convolutional blocks within CNN by using "blocks" argument while calling the main script or numer of units within different types of recurrent networks (Simple RNN, LSTM, GRU). You can also specify callbacks, which can adjust learing rate, early stopping or model checkpoint. After training model is evaluated by model_summary.py script containing methods returning plots of training/testing accuracy, training/testing loss, classification report, and confusion matrix (in form of a heat map). If validation argument is set as true model runs through every file in data predicting its class and then returns list of IDs of images, which were misscalssified. 

# GAN image superresolution
This directory contains functions dedicated for trainig generative adversarial neural network (GAN) in order to recreate high-resolution imaged based on low-resolution ones, having also original high-resolution images for validation purposes. In this case high-resolution (ground-truth) images are derrived from confocal microscopy and low-resolution are derrived from wide-filed fluorescence microscopy. Main script, which can be called with appropriate arguments (described in the file) within terminal is called train_GAN.py. First Program loads high- and low-resolution images, assigns appropriate weigths to the loss functions in generator (Binary cross-entropy, mean squared error and structural similarity index). Later, both discriminator (build as a simple CNN) and generator (build based on the U-NET architecture) are defined and the GAN is trained. The model is saved based on the metrics like MSE, NRMSE (normalized MSE), SSIM, and PSNR (peak signal-to-noise ratio). Best model can be later read-in via predictions.py script, which returns plots containig HR (ground truth), LR and generated samples, cross-section of the image along given pixel value and described earlier metrics.
