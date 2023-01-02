import tensorflow as tf
from .base import ModelBuilder


class HighwayModel(ModelBuilder):
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
            tf.keras.layers.Dense(self.nb_units[0], activation='relu'),
        ])
        for nb_unit in self.nb_units:
            model.add(Highway(activation='relu', tgBias=-3.))
        model.add(tf.keras.layers.Dense(self.classes))
        return model


class Highway(tf.keras.layers.Layer):

    def __init__(self, activation=None, tgBias=-1.):
        super(Highway, self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.tgActivation = tf.keras.activations.get('sigmoid')
        self.tgBias_init = tgBias

    def build(self, input_shape):
        dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", shape=[dim, dim])
        self.bias = self.add_weight("bias", shape=[dim,])
        self.tgKernel = self.add_weight("tgKernel", shape=[dim,dim])
        self.tgBias = self.add_weight("tgBias", shape=[dim], initializer=tf.keras.initializers.Constant(self.tgBias_init))
        self.built = True

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel)
        outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        transform_gate = tf.matmul(inputs, self.tgKernel)
        transform_gate = tf.nn.bias_add(transform_gate, self.tgBias)
        transform_gate = self.tgActivation(transform_gate)
        outputs = tf.math.multiply(outputs, transform_gate) + tf.math.multiply(inputs, tf.math.add(1.,-transform_gate))
        return outputs