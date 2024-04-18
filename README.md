# Real-Time Digit Recognition with TensorFlow and OpenCV

This Python script utilizes TensorFlow and OpenCV for real-time digit recognition from a webcam feed. It trains a Convolutional Neural Network (CNN) model using the MNIST dataset and then predicts digits in real-time from the video stream.

## Features

- **Real-Time Recognition:** Utilizes OpenCV to capture video from a webcam and performs real-time digit recognition.
- **CNN Model Training:** Trains a CNN model on the MNIST dataset to recognize handwritten digits.
- **User Interaction:** Allows users to toggle between inference mode and normal video display with a mouse click.
- **Threshold Adjustment:** Provides a trackbar interface to adjust the threshold for image binarization in real-time.

## Dependencies

- TensorFlow
- NumPy
- OpenCV

## Usage

1. Install the required dependencies (`tensorflow`, `numpy`, `opencv-python`).
2. Run the `real_time_digit_recognition.py` script.
3. Point the webcam towards handwritten digits to see real-time recognition.
