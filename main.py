# Install Dependencies and Setup
import numpy as np
import cv2  # OpenCV
import sklearn  # Scikit-Learn
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load Dataset
dataset_path = 'face_dataset/lfw-deepfunneled'

# Initialize a list to store the image data and labels
images = []
labels = []

# Walk through the directory to list all the subdirectories and files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            # Load the image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Append the image and its label
            images.append(image)
            labels.append(root.split(os.sep)[-1])  # The label is the folder name

# Convert the lists to NumPy arrays if needed
images_np = np.array(images)
labels_np = np.array(labels)

print(labels[1])
(images[1])

import matplotlib.pyplot as plt
import random


# Function to display images
def show_images(images, labels, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        # Pick a random index for sampling images
        idx = random.randint(0, len(images) - 1)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx], cmap='gray')  # Show the image in grayscale
        plt.title(labels[idx])  # The title is the label of the image
        plt.axis('off')  # Turn off the axis
    plt.show()


# Image Preprocessing
def preprocess_image(image_array):
    # Resize the image to 224x224 pixels
    resized_image = cv2.resize(image_array, (224, 224))  # Using OpenCV for resizing
    # Normalize the image
    normalized_image = resized_image / 255.0
    return normalized_image


# Preprocess all images
preprocessed_images = [preprocess_image(img) for img in images]
# print(len(preprocessed_images))

# If you want to visualize the preprocessed images
show_images(preprocessed_images, labels)

# Split the data
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming you have your data (images and labels) stored in `images_np` and `labels_np` NumPy arrays

# Define the ratios for splitting (e.g., 70% training, 15% validation, 15% test)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images_np, labels_np, test_size=(1 - train_ratio), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio),
                                                random_state=42)

# Check the sizes of the resulting sets
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# Show random images from the training set
print("Training Dataset")
show_images(X_train, y_train, num_images=5)

# Show random images from the validation set
print("Validation Dataset")
show_images(X_val, y_val, num_images=5)

# Show random images from the test set
print("Test Dataset")
show_images(X_test, y_test, num_images=5)

image_shape = images_np.shape

# Print the shape to confirm if it's a 2D array
print("Shape of images_np:", image_shape)

# CNN model
import numpy as np

kernel1 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])  # Edge detection kernel

kernel2 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])  # Sharpening kernel


def convolution2d(image, kernel):
    """
    Perform 2D convolution between an image and a kernel.

    Args:
    image (numpy.ndarray): Input grayscale image (height x width).
    kernel (numpy.ndarray): Convolution kernel (k_height x k_width).

    Returns:
    numpy.ndarray: Convolved image.
    """
    if len(image.shape) != 2 or len(kernel.shape) != 2:
        raise ValueError("Both image and kernel should be 2D arrays.")

    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2  # Padding to maintain output size

    # Initialize the output image
    output = np.zeros((h, w))

    # Perform convolution
    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            region = image[i - pad_h: i + pad_h + 1, j - pad_w:  j + pad_w + 1]
            output[i, j] = np.sum(region * kernel)

    return output


def relu(x):
    """
    ReLU activation function.

    Args:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying ReLU.
    """
    return np.maximum(0, x)


def max_pooling2d(image, pool_size=(2, 2)):
    p_h, p_w = pool_size
    h, w = image.shape
    pooled_h = h // p_h
    pooled_w = w // p_w

    pooled_image = np.zeros((pooled_h, pooled_w))

    for i in range(pooled_h):
        for j in range(pooled_w):
            region = image[i * p_h:(i + 1) * p_h, j * p_w:(j + 1) * p_w]
            pooled_image[i, j] = np.max(region)

    return pooled_image

def apply_two_layer_convolution(image):
    """
    Apply two layers of convolution, ReLU, and max pooling to an image.

    Args:
    image (numpy.ndarray): The input grayscale image.

    Returns:
    numpy.ndarray: The output image after two layers of processing.
    """
    # First layer
    conv1 = convolution2d(image, kernel1)
    relu1 = relu(conv1)
    pool1 = max_pooling2d(relu1)

    # Second layer
    conv2 = convolution2d(pool1, kernel2)
    relu2 = relu(conv2)
    pool2 = max_pooling2d(relu2)

    return pool2


def transform_dataset_with_convolution(dataset):
    """
    Apply the two-layer convolution process to each image in the dataset.

    Args:
    dataset (numpy.ndarray): A batch of images.

    Returns:
    numpy.ndarray: The transformed batch of images.
    """
    # Initialize a list to hold the transformed images
    transformed_images = []

    # Loop over each image in the dataset
    for image in dataset:
        # Apply the two-layer convolution process
        transformed_image = apply_two_layer_convolution(image)
        # Append the transformed image to our list
        transformed_images.append(transformed_image)

    # Convert the list back to a NumPy array
    return np.array(transformed_images)


# Apply the transformation to each of your datasets
# Assuming the function transform_dataset_with_convolution is already defined and works as intended

# Apply the transformation to the first 100 images of each dataset
X_train_transformed_100 = transform_dataset_with_convolution(X_train[:100])
X_val_transformed_100 = transform_dataset_with_convolution(X_val[:100])
X_test_transformed_100 = transform_dataset_with_convolution(X_test[:100])

# Now, X_train_transformed, X_val_transformed, and X_test_transformed contain the images
# after undergoing two layers of convolution, ReLU, and max pooling operations.

import matplotlib.pyplot as plt


def visualize_cnn_steps(images, titles):
    """
    Visualize a list of images in a single row with their respective titles.

    Args:
    images (list of numpy.ndarray): The images to be visualized in a single row.
    titles (list of str): Titles for each image.
    """
    n = len(images)  # Number of images (and titles)
    plt.figure(figsize=(n * 4, 4))  # Adjust the figure size as needed

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 1)  # 1 row, n columns, ith subplot
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


def apply_and_visualize_cnn_operations_for_selected_images(images, kernel1, kernel2, pool_size=(2, 2),
                                                           visualize_indices=[0, 1, 2, 3, 4]):
    """
    Apply two sets of convolution, ReLU activation, and max pooling to selected images
    and visualize the results for each selected image.
    """
    for idx in visualize_indices:
        image = images[idx]
        print(f"Visualizing operations for Image {idx + 1}: 1st Convolutional Layer")
        # First Convolutional Layer Operations
        convolved1 = convolution2d(image, kernel1)
        activated1 = relu(convolved1)
        pooled1 = max_pooling2d(activated1, pool_size)
        visualize_cnn_steps([image, convolved1, activated1, pooled1],
                            ["Original", "1st Convolution", "1st ReLU", "1st Pooling"])

        print(f"Visualizing operations for Image {idx + 1}: 2nd Convolutional Layer")
        # Second Convolutional Layer Operations
        convolved2 = convolution2d(pooled1, kernel2)
        activated2 = relu(convolved2)
        pooled2 = max_pooling2d(activated2, pool_size)
        visualize_cnn_steps([image, convolved2, activated2, pooled2],
                            ["Original", "2nd Convolution", "2nd ReLU", "2nd Pooling"])


# Apply the CNN operations and visualize the process for the first 5 images
apply_and_visualize_cnn_operations_for_selected_images(images, kernel1, kernel2, visualize_indices=[7, 11, 31, 51, 81])


def categorical_cross_entropy_loss(predictions, true_labels):
    """
    Calculate the categorical cross-entropy loss.

    Args:
        predictions (numpy.ndarray): The predictions from the model, shape (n_samples, n_classes), probabilities.
        true_labels (numpy.ndarray): The true labels, one-hot encoded, shape (n_samples, n_classes).

    Returns:
        float: The average categorical cross-entropy loss over all samples.
    """
    # Small value to avoid log(0)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # Calculate the loss for each sample and class
    loss = -np.sum(true_labels * np.log(predictions)) / predictions.shape[0]
    return loss


# Step 1: Initialize a list to store the loss values
loss_values = []

# This is a simplified loop to simulate the training process
for epoch in range(1, 101):  # Let's say you are training for 100 epochs
    # Step 2: Calculate loss here (this is a dummy value for the sake of example)
    loss = 1 / epoch  # Dummy example to simulate decreasing loss

    # Record the loss value
    loss_values.append(loss)

    # Optionally, print the loss every few epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Step 3: Plot the loss values
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()