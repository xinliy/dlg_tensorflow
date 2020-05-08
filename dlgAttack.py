import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import numpy as np
import pprint


def one_hot_labels(y, n_class=10):
    r = np.zeros((y.size, n_class))
    r[np.arange(y.size), y] = 1
    return r


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.d2 = Dense(10, input_shape=(784,), activation='sigmoid')

    def call(self, x):

        return self.d2(x)


# @tf.function
def get_original_dy_dx(model, gt_data, gt_label, loss_object):
    """Get the orignial gradients from model and pair of ground truth data."""
    with tf.GradientTape() as g:
        pred = model(gt_data)
        loss = loss_object(pred, gt_label)
    origin_dy_dx = g.gradient(loss, model.trainable_variables)
    return origin_dy_dx


def train_dummy(model, gt_data, gt_label, dummy_x, dummy_y, loss_object):
    """Recontruct dummy data in one epoch."""
    mse = tf.keras.losses.MeanSquaredError()
    op = tf.keras.optimizers.Adagrad(learning_rate=0.2)
    # TODO: Try lbfgs optimizer

    with tf.GradientTape(persistent=True) as g:
        g_diff = 0

        with tf.GradientTape() as gg:
            origin_dy_dx = get_original_dy_dx(
                model=model, gt_data=gt_data, gt_label=gt_label, loss_object=loss_object)
            dummy_output = model(dummy_x)
            # TODO: Cannot add a softmax function to dummy_y
            loss = loss_object(dummy_output, dummy_y)
        dummy_g = gg.gradient(loss, model.trainable_variables)

        for l1, l2 in zip(origin_dy_dx, dummy_g):
            g_loss = mse(l1, l2)
            g_diff += g_loss

    update_g = g.gradient(g_diff, [dummy_x, dummy_y])
    op.apply_gradients(zip(update_g, [dummy_x, dummy_y]))
    return float(g_diff)


def DLG_Attack(model, gt_data, gt_label, n_iter=300, stop_loss=1e-4, verbose=0):
    """Implement Deep Leakage from Gradients (DLG) attack algorithm.

    Args:
        model: A tf model
        gt_data: One sample ground truth data.
        gt_label: Ground Truth label in one-hot form.
        n_iter: Number of iterations to train.
        stop_loss: Stop train if reach this loss.
        verbose: print training info in a given interval.
    Returns:
        A pair of results.

    """
    dummy_x = tf.Variable(tf.random.normal(
        gt_data.shape, mean=0.01, stddev=0.001))
    dummy_y = tf.Variable(tf.random.normal(
        gt_label.shape, mean=0.01, stddev=0.001))
    crossentropy = tf.keras.losses.CategoricalCrossentropy()

    for i in range(n_iter):
        loss = train_dummy(model=model, gt_data=gt_data,
                           gt_label=gt_label, dummy_x=dummy_x, dummy_y=dummy_y, loss_object=crossentropy)

        if verbose and (i + 1) % verbose == 0:
            print("Iter:{}|loss:{:.5f}".format(i + 1, loss))
        if loss < stop_loss:
            print("Break at iter {} with loss:{:.5f}".format(i + 1, loss))
            break
    return dummy_x, dummy_y
