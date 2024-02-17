# Face Recognition with Convolutional Neural Networks (CNN) 

## Project Description
This Python project demonstrates the basics of face recognition using Convolutional Neural Networks (CNN), 
leveraging popular libraries such as OpenCV for image processing, NumPy for numerical operations, Scikit-Learn 
for machine learning tasks, and Matplotlib for visualization. The script loads a dataset of faces, preprocesses 
the images (including resizing and normalization), and then splits the data into training, validation, and test 
sets. It also includes a simple implementation of 2D convolution, ReLU activation, and max pooling operations to
illustrate the foundational components of CNNs.

## Features
* Image Loading and Preprocessing: Utilizes OpenCV to load and preprocess images for neural network training, including grayscale conversion, resizing, and normalization.
* Dataset Splitting: Employs Scikit-Learn to divide the dataset into training, validation, and test sets according to specified ratios.
* CNN Components: Contains simple Python implementations of key CNN operations, such as 2D convolution, ReLU activation, and max pooling.
* Visualization: Uses Matplotlib to display preprocessed images and visualize the dataset distribution across training, validation, and test sets.


## Installation
Ensure you have Python installed on your system (preferably Python 3.x). You will also need to install the required dependencies. 

You can install all the necessary libraries using pip:

bash

pip install numpy opencv-python scikit-learn matplotlib pandas

## Dataset
The dataset should be organized in a directory named face_dataset/lfw-deepfunneled, where each subdirectory represents a class 
(person) and contains images of that person's face. The script assumes the dataset is in this format for loading and preprocessing.


