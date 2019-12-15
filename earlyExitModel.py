import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from earlyExitDenseBlock import EarlyExitDenseBlock

class EarlyExitModel(keras.Model):
    def __init__(self):
        super(EarlyExitModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.ed1 = EarlyExitDenseBlock(128)
        self.ed2 = EarlyExitDenseBlock(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x, y1 = self.ed1(x)
        x, y2 = self.ed2(x)
        # x = keras.layers.concatenate([x, y1, y2])
        return x