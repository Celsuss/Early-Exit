import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class EarlyExitDenseBlock(keras.Model):
    def __init__(self, size, activation='relu'):
        super(EarlyExitDenseBlock, self).__init__()

        self.dense1 = layers.Dense(size)
        self.dense2 = layers.Dense(10)
        if activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'softmax':
            self.activation = tf.nn.softmax

    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        x = self.activation(x)
        y = self.dense2(x)
        y = tf.nn.softmax(y)
        return x, y