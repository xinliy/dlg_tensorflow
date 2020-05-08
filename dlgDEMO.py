import tensorflow as tf
import matplotlib.pyplot as plt
from dlgAttack import *


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[:1]
    x_train = x_train.reshape(1, -1)
    y_train = y_train[:1]

    gt_data = x_train
    gt_label = one_hot_labels(y_train)
    print("true shape:", gt_data.shape, gt_label.shape)
    model = MyModel()

    dummy_x, dummy_y = DLG_Attack(
        model=model, gt_data=gt_data, gt_label=gt_label, verbose=10)

    faker = dummy_x.numpy().reshape(28, 28)
    plt.imshow(faker)
    plt.show()
