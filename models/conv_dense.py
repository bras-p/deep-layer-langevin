import tensorflow as tf
from .base import ModelBuilder


class ConvDense(ModelBuilder):
    def __init__(
        self,
        filters = 32,
        kernel_size = 4,
        nb_conv = 1,
        nb_units = 2*[64],
        classes = 10,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.nb_units = nb_units
        self.classes = classes

    def getModel(self):
        model = tf.keras.Sequential([])
        for _ in range(self.nb_conv):
            model.add(tf.keras.layers.Conv2D(self.filters, self.kernel_size, activation='relu'))
            model.add(tf.keras.layers.MaxPool2D((2,2)))
        model.add(tf.keras.layers.Flatten())
        for nb_unit in self.nb_units:
            model.add(tf.keras.layers.Dense(nb_unit, activation='relu'))
        model.add(tf.keras.layers.Dense(self.classes))
        return model