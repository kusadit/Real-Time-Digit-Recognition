# Real-Time Digit Recognition

This project implements a real-time digit recognition system using a pre-trained neural network on the MNIST dataset and OpenCV for live video processing. The program allows users to draw digits in a specified region of the webcam feed and recognizes them using the trained model.

## Features
- **Real-time digit recognition**: Detects and recognizes digits drawn in the live webcam feed.
- **Dynamic threshold adjustment**: Adjust the threshold value for better image preprocessing using a slider.
- **Interactive controls**: Start and stop digit inference with mouse clicks.
- **Training capability**: Automatically trains the model if no pre-trained model is found.

## Prerequisites
- Python 3.7+
- TensorFlow
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kusadit/real-time-digit-recognition.git
   cd real-time-digit-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the MNIST dataset. If not, the script will download it automatically when training.

## Usage

### Run the Application
To start the real-time digit recognition:
```bash
python AI.py
```

### Key Features
- **Mouse Interaction**:
  - Click anywhere on the webcam feed to toggle digit inference on or off.
- **Threshold Adjustment**:
  - Use the threshold slider to adjust the binary threshold value for preprocessing.
- **Exit**:
  - Press the `q` key to exit the application.

### Model Training
If a pre-trained model (`model_saved.h5`) is not found, the script will:
1. Download the MNIST dataset.
2. Train a neural network on the dataset.
3. Save the trained model for future use.

## Code Overview

### Key Components

#### `train_model(x_train, y_train, x_test, y_test)`
Trains a neural network on the MNIST dataset and stops training early if accuracy exceeds 99%.

#### `load_mnist_data()`
Loads the MNIST dataset for training and testing.

#### `predict(model, img)`
Uses the trained model to predict the digit in a given image.

#### `start_cv(model)`
Handles live webcam feed, processes frames, and performs digit recognition in real time.

### Constants
- `MODEL_FILE`: Path to the saved model file.
- `MNIST_PATH`: Path to the MNIST dataset.
- `THRESHOLD_DEFAULT`: Default threshold value for image preprocessing.
- `FRAME_COUNT_RESET`: Number of frames before resetting inference.

## Requirements
The following Python libraries are required to run this project:
- TensorFlow
- NumPy
- OpenCV

Install them via `pip`:
```bash
pip install tensorflow opencv-python-headless numpy
```

## File Structure
- `AI.py`: Main script for running the application.
- `README.md`: Documentation file.
- `model_saved.h5`: Pre-trained model file (if available).

## Notes
- Ensure your webcam is connected and working correctly before running the script.
- Adjust the threshold slider for better recognition accuracy depending on lighting conditions and input.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The MNIST dataset is provided by [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/).
- TensorFlow and OpenCV are used for model training and computer vision tasks.

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to suggest improvements.




