import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Trains CNN Model for set # epochs over training img xray directory
# XRAY PNEUMONIA IMAGE DATASET FROM: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
def train(model, epochs, load_cp):
    img_width = 180
    img_height = 180

    # Training Data directory
    data_dir = "data/chest_xray/train/"
    data_path = pathlib.Path(data_dir)

    image_count = len(list(data_path.glob('*/*.jpeg')))
    normal = list(data_path.glob('NORMAL/*'))

    batch_size = 32

    # Get training dataset from directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.15,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Separate validation dataset, 85% train - 15% val
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.15,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    # Optimize training buffer with Tensorflow AUTOTUNE
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Create checkpoint
    checkpoint_path = "training_checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    if load_cp:
        model.load_weights(checkpoint_path)

    # Training loop saved as history, saves checkpoint every epoch
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=[cp_callback]
    )

    generate_figure(history, epochs)

# Uses training loop history to generate matplotlib graph with accuracy/loss x epoch #
# Visualizing training history: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def generate_figure(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Runs batch of 1 image to CNN model predict function, returns tuple of (classification, confidence score)
def predict_img(model, path):
    checkpoint_path = "training_checkpoints/cp.ckpt"
    width = 180
    height = 180
    classes = ['NORMAL', 'PNEUMONIA']

    # Load trained model weights from checkpoint
    model.load_weights(checkpoint_path)

    # Get & resize image to 180 x 180
    img = tf.keras.preprocessing.image.load_img(
        path, target_size=(width, height)
    )

    # Convert image to correct # dimension array to input to model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Model predicts diagnoses of img array, get label and confidence score
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    label = classes[np.argmax(score)]
    confidence = float(100 * np.max(score))

    print(predictions)
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(classes[np.argmax(score)],
                                                                                           100 * np.max(score)))
    # Returns to rendered results.html page
    return label, confidence


# ~10 % of original 2090 x 1858 training images
img_width = 180
img_height = 180

# Create more data from training directory to combat overfitting, horizontal flip
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
  ]
)


num_classes = 2

# Keras-TF CNN Model with Convolutional, Max Pooling, Dropout, and Dense layer(s)
# Using recommended params & RELU activation function
model = Sequential([
  data_augmentation,
  # normalize array nums between 0 and 1 instead of 0-255 for RGB
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Define loss function, optimizer, and metrics of CNN model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
