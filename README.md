# Image-classification-using-CNN

COMPANY: CODTECH IT SOLUTIONS

NAME: SREEMATHI.R

INTERN ID: CT06DH682

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH
 📌 Project Overview
This project was developed as part of **CODTECH Internship Task-3** with the goal of creating a functional image classification system using a Convolutional Neural Network (CNN). CNNs are a class of deep learning models specifically designed for processing visual data like images. This model is trained and tested on the **CIFAR-10 dataset**, which is one of the most widely used benchmark datasets in the computer vision domain.

The CIFAR-10 dataset consists of 60,000 32x32 color images categorized into 10 classes, including vehicles (cars, trucks, airplanes, etc.) and animals (cats, dogs, deer, etc.). It has 50,000 training images and 10,000 test images.

🎯 Objectives

- Build a convolutional neural network using TensorFlow/Keras.
- Train the model on the CIFAR-10 image dataset.
- Evaluate the model's performance on test data.
- Learn and apply basic computer vision techniques using deep learning.

⚙️ Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Matplotlib (optional for visualization)
  
🛠️ How the Model Works

1. Dataset Loading: The CIFAR-10 dataset is loaded using `tf.keras.datasets.cifar10`, which automatically splits the data into training and test sets.
2. Preprocessing: The pixel values of the images are normalized by dividing by 255 to bring them into a range of 0 to 1, which helps in faster convergence.
3. CNN Architecture:
    - 3 Convolutional layers with increasing filters (32, 64, 64)
    - MaxPooling layers to reduce spatial dimensions
    - Flatten layer to convert the 2D matrix into a 1D vector
    - Dense layers for classification with the final layer outputting logits for 10 classes
4. Compilation: The model uses the Adam optimizer and sparse categorical crossentropy loss, which is suitable for multi-class classification.
5. Training: The model is trained over 10 epochs and validated using the test dataset.
6. Evaluation: After training, the test accuracy is calculated and displayed.

📊 Results

The CNN achieves approximately **70–75% accuracy** on the test dataset after 10 epochs. This demonstrates the model's ability to learn from the CIFAR-10 dataset and perform basic classification tasks effectively.
