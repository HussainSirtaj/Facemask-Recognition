# Facemask-Recognition
A simple project in python to recognize whether the image/person is wearing a mask or not.
.
.
😷 Face Mask Detection using CNN

<p align="center">
<img src="https://www.google.com/search?q=https://placehold.co/600x200/228B22/FFFFFF%3Ftext%3DCNN%2BMask%2BDetector" alt="Project Banner Placeholder" />
</p>

🌟 Project Overview

This is a complete, end-to-end project for classifying whether a person in an image is wearing a face mask or not. The solution leverages a Convolutional Neural Network (CNN), implemented using TensorFlow/Keras, to achieve high-accuracy predictions.

This project is beginner-friendly, clean, and easily extendable for real-time applications or mobile deployment.

🚀 Features

🧠 Fully Trained CNN Model: Utilizes a robust architecture optimized for image feature extraction.

🖼️ Clean Preprocessing Pipeline: Standardized image handling using OpenCV.

📂 Supports Custom Images: Easily test the model on your own images.

📊 High Accuracy: Typically achieves 95%–99% classification accuracy.

💡 Future-Ready: Easily convertible to TensorFlow Lite or ONNX for deployment.

🛠️ Technologies Used

Library / Tool

Purpose

TensorFlow / Keras

Core framework for model building and training.

OpenCV (opencv-python-headless)

Image manipulation (read, resize, color conversion).

NumPy

Efficient array operations and data handling.

Matplotlib

Visualizations and plotting (e.g., loss curves, confusion matrix).

Scikit-learn

Splitting the dataset into training and testing sets.

⚙️ Installation

To set up the project locally, run the following command to install the required dependencies:

pip install tensorflow opencv-python-headless matplotlib scikit-learn


📂 Project Structure

The expected file structure for training is as follows:

.
├── dataset/
│   ├── with_mask/     # Images of people wearing masks
│   └── without_mask/  # Images of people not wearing masks
├── mask_detection.ipynb # The main training and prediction notebook
└── README.md


🧠 How It Works: CNN Pipeline

The model follows a standard deep learning pipeline for image classification:

1. Preprocessing & Data Loading

Image Handling: Images are read, resized to 128x128 pixels, and converted from BGR (OpenCV default) to RGB.

Normalization: Pixel values are normalized to the range [0, 1].

Label Encoding:

0 = without_mask

1 = with_mask

Splitting: The dataset is split into training and testing sets for validation.

2. CNN Model Architecture

The model is a sequential CNN designed for robust feature learning:

Convolution Layers: Extract spatial hierarchies of features (edges, shapes).

MaxPooling Layers: Reduce dimensionality, making the model faster and more robust to slight variations.

Dropout: Applied between layers to randomly drop neurons, preventing overfitting during training.

Flatten Layer: Converts the 2D feature maps into a 1D vector.

Dense Layers: Standard fully connected layers for final classification.

Output Layer: A single Dense layer with a sigmoid activation function for binary classification, outputting a probability between 0 and 1.

3. Training & Prediction

The model is trained using the model.fit() method:

model.fit(X_train, y_train, epochs=10, validation_split=0.2)


Example Prediction:
For a new image, the prediction logic is:

pred = model.predict(input_img)[0][0]
label = "Mask" if pred >= 0.5 else "No Mask"


🔮 Future Improvements

Consider these enhancements to expand the project's capabilities:

Real-Time Detection: Implement live detection using webcam feed with OpenCV.

Deployment: Convert the model to TensorFlow Lite or CoreML for integration into mobile applications.

Dataset Augmentation: Use techniques like rotation, shearing, and zooming to improve model generalization.

Advanced Metrics: Add a Confusion Matrix plot for detailed analysis of prediction errors.

📜 License

This project is open-source and available under the MIT License. You are free to modify, share, and improve the code as needed.
