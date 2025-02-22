import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional, Reshape
import numpy as np
import os

base_path = "data"

# Load words
def load_data():
    with open(f"{base_path}/words.txt", "r") as f:
        words = f.readlines()
    words_list = [word.strip() for word in words]
    return words_list

words_list = load_data()
split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

# Create character mapping
characters = set()
for word in words_list:
    characters.update(word)
characters = sorted(characters)

char_to_num = keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Define model
def build_model(image_width, image_height):
    input_img = Input(shape=(image_width, image_height, 1), name="image")
    x = Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Reshape((-1, 128))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    output = Dense(len(characters) + 1, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=input_img, outputs=output)
    return model

# Define loss function
def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.ones(shape=(batch_len, 1), dtype="int64") * tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.ones(shape=(batch_len, 1), dtype="int64")
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Compile and summarize model
image_width, image_height = 128, 32
model = build_model(image_width, image_height)
model.compile(optimizer="adam", loss=ctc_loss)
model.summary()
