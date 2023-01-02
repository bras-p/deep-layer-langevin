import tensorflow as tf
from .base import ModelBuilder


class DenseNet(ModelBuilder):
    def __init__(
        self,
        input_shape = (32,32,2,),
        classes = 10,
    ):
        self.input_shape = input_shape
        self.classes = classes

    def getModel(self):
        model = tf.keras.applications.densenet.DenseNet121(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=self.input_shape,
            pooling=None,
            classes=self.classes,
            classifier_activation=None
        )
        return model


