import tensorflow as tf
from .base import ModelBuilder


class DenseModel(ModelBuilder):
    def __init__(
        self,
        nb_units = 2*[64],
        classes = 10,
    ):
        self.nb_units = nb_units
        self.classes = classes

    def getModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
        ])
        for nb_unit in self.nb_units:
            model.add(tf.keras.layers.Dense(nb_unit, activation='relu'))
        model.add(tf.keras.layers.Dense(self.classes))
        return model