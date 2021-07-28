import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import RMSprop


class Preprocessor():
    def __init__(self):
        print("Preprocessor initialized...")

    def preprocess(self, image):
        normalized = image / 255.
        gray_scaled = tf.image.rgb_to_grayscale(normalized)
        resized = tf.image.resize(gray_scaled, np.array([84, 84]))
        reshaped = np.reshape(resized, (84, 84, 1))
        return reshaped


class ImageModelHandler:
    def __init__(self):
        print("Image Model initialized...")

    def createModel(self, input_shape: tuple, output_shape: int):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, 8, strides=4,
                       padding='same', activation="relu"),
                Conv2D(64, 4, strides=2,
                       padding='same', activation="relu"),
                Conv2D(64, 4, strides=1,
                       padding='same', activation="relu"),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(output_shape, activation='linear')
            ]
        )
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0)
                      )
        return model

    def updateModel(self, mainModel, targetModel) -> bool:
        targetModel.set_weights(mainModel.get_weights())
        return True

    def load(self, name):
        loaded = tf.keras.models.load_model(name)
        return loaded

    def save(self, name: str, model) -> bool:
        model.save(name)
        return True
