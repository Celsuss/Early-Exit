from __future__ import absolute_import, division, print_function, unicode_literals  

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from refModel import RefModel
from earlyExitModel import EarlyExitModel

print('Tensorflow version {}'.format(tf.__version__))

TRAIN_BATCH_COUNT = 0

def getMnistDatasets(batch_size=32):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    global TRAIN_BATCH_COUNT
    TRAIN_BATCH_COUNT = int(len(y_train) / batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds

# @tf.function
def trainStep(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# @tf.function
def testStep(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

def trainModel(model, train_ds, test_ds, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, epochs=5):
    for epoch in range(epochs):
        train_step_count = 0
        for images, labels in train_ds:
            trainStep(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
            train_step_count += 1
            print('Train step {}/{}'.format(train_step_count, TRAIN_BATCH_COUNT), end='\r')

            # TODO: KEEP FOR TESTING ONYL
            if train_step_count >= 10:
                break

        for test_images, test_labels in test_ds:
            testStep(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

        template = '[Epoch {}] Loss: {:.3f}, Accuracy: {:.2%}, Test Loss: {:.3f}, Test Accuracy: {:.2%}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result(),
                                test_loss.result(),
                                test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    return model

def main():
    train_ds, test_ds = getMnistDatasets(32)

    # Create an instance of the model
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    ref_model = RefModel()

    epochs = 5
    tf.random.set_seed(10)
    # ref_model = trainModel(ref_model, train_ds, test_ds, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, epochs)

    model = EarlyExitModel()
    model = trainModel(model, train_ds, test_ds, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, epochs)

    

if __name__ == '__main__':
    main()
    print("Exiting")