import tensorflow as tf
import numpy as np
import cv2
import sys

# Constants
MODEL_FILE = 'model_saved.h5'
MNIST_PATH = 'mnist.npz'
THRESHOLD_DEFAULT = 150
FRAME_COUNT_RESET = 5

# Change console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Train model with MNIST data
def train_model(x_train, y_train, x_test, y_test):
    # Callback to stop training when accuracy reaches 99%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy. Cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit model
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    print(history.epoch, history.history['accuracy'][-1])
    return model

# Load MNIST data
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=MNIST_PATH)
    return x_train, y_train, x_test, y_test

# Predict digit using image
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# OpenCV functions

# Mouse click event handler
startInference = False
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Threshold slider handler
threshold = THRESHOLD_DEFAULT
def on_threshold(x):
    global threshold
    threshold = x

# Main function to start OpenCV
def start_cv(model):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', THRESHOLD_DEFAULT, 255, on_threshold)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if startInference:
            frameCount += 1

            # Convert to grayscale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            # Get central image
            resizedFrame = thr[240-75:240+75, 320-75:320+75]

            # Resize for inference
            iconImg = cv2.resize(resizedFrame, (28, 28))

            # Pass to model predictor
            res = predict(model, iconImg)

            # Clear background periodically
            if frameCount == FRAME_COUNT_RESET:
                frameCount = 0

            # Display result
            cv2.putText(frame, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 225, 255), 3)
            cv2.rectangle(frame, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=8)

            cv2.imshow('background', frame)
        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model = None
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print('Loaded saved model.')
        print(model.summary())
    except (OSError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
        print(f"Error loading model: {e}")
        print("Getting MNIST data...")
        try:
            x_train, y_train, x_test, y_test = load_mnist_data()
        except Exception as e:
            print("Error getting MNIST data:")
            print(repr(e).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            return
        print("Training model...")
        model = train_model(x_train, y_train, x_test, y_test)
        print("Saving model...")
        model.save(MODEL_FILE)

    print("Starting CV...")
    start_cv(model)

# Call main
if __name__ == '__main__':
    main()
