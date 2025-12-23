# Deep Learning on Fashion-MNIST

This notebook documents my practical work on the **Fashion-MNIST dataset** while learning deep learning fundamentals. The focus of this work is to understand how a neural network learns from image data by building, training, and evaluating a model step by step.

---

## Project File

- Deep_Learning.ipynb – Notebook containing the full implementation and observations

---

## 1. Goal of the Work

The goal of this notebook is to:
- Work with an image-based dataset
- Understand how neural networks handle pixel data
- Build a basic deep learning model from scratch
- Observe model performance through training and evaluation

---

## 2. Dataset – Fashion-MNIST

The dataset used is **Fashion-MNIST**, which contains:
- 70,000 grayscale images of clothing items
- 10 classes such as T-shirt, Trouser, Dress, Sneaker, Bag, etc.
- Image size of 28 × 28 pixels

The dataset is split into training and test sets.

---

## 3. Loading and Exploring the Dataset

Steps performed:
- Load the Fashion-MNIST dataset using a deep learning library
- Check the shape of training and test data
- Inspect labels
- Visualize sample images along with their class labels

This helped understand what the model is expected to learn.

---

## 4. Data Preprocessing

Before feeding data to the model:
- Pixel values are scaled (normalized) to improve training
- Images are reshaped/flattened to match the input layer
- Training and test datasets are prepared properly

Normalization was necessary for stable and faster learning.

---

## 5. Building the Neural Network

A **basic feedforward neural network** is created:
- Input layer corresponding to image pixels
- One or more hidden dense layers
- Activation functions applied to hidden layers
- Output layer with Softmax activation for multi-class classification

This model represents the first deep learning approach applied to the dataset.

---

## 6. Model Compilation

The model is compiled by defining:
- Loss function suitable for multi-class classification
- Optimizer to update weights
- Accuracy as the evaluation metric

This prepares the model for training.

---

## 7. Model Training

The neural network is trained using:
- Fashion-MNIST training images and labels
- Multiple epochs
- Monitoring loss and accuracy during training

Training results show how the model gradually improves its predictions.

---

## 8. Model Evaluation

After training:
- The model is evaluated on the test dataset
- Test accuracy is calculated
- Model performance on unseen images is analyzed

This step confirms how well the network generalizes.

---

## 9. Observations

- Normalizing pixel values improves convergence
- Neural networks can learn meaningful patterns from image data
- Model accuracy improves with training epochs
- Simple dense networks can classify Fashion-MNIST reasonably well

---

## 10. Conclusion

This notebook represents my first practical deep learning implementation on image data. By working with the Fashion-MNIST dataset, I learned how neural networks process images, how preprocessing affects learning, and how model training and evaluation work in practice.

---

