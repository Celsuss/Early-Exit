import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class EarlyExitDenseBlock(keras.Model):
    def __init__(self):
        super(EarlyExitDenseBlock, self).__init__()

        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        y1 = tf.nn.softmax(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return tf.nn.softmax(x), y1